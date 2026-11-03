"""Infrastructure for the Telecom Researcher (AWS CDK, Python), deployed in cost phases.

  cdk deploy -c phase=1   CORE (default, low cost, pay per use): field + lake buckets, DynamoDB findings/audit/truth,
                          run_experiment and lab_execute Lambdas (one container image: MAVERIC twin + emulated lab) for
                          AgentCore Gateway, SQS job queue with a nightly EventBridge rule, the agents' IAM role
  cdk deploy -c phase=2   + full pipeline: Kinesis streams, Glue tables + Athena, Step Functions protocol with its task
                          Lambda and an EventBridge Scheduler nightly run, review console (Cognito, API Gateway + Lambda,
                          Amplify Hosting), SageMaker training role, Bedrock Custom Model Import role, model bucket
  cdk deploy -c phase=3   + always-on services: VPC with a NAT gateway, Neptune serverless finding graph, OpenSearch
                          Serverless collection, Managed Grafana (needs IAM Identity Center). Phases 1-2 keep the finding
                          graph and log search local inside the Lambdas, at no extra cost.

The AgentCore Gateway and Runtime are created by deploy/provision_gateway.py and `agentcore launch`.
STATUS: `cdk synth` succeeds for every phase; not deployed yet.
"""
import json
import sys
from pathlib import Path

import jsii
from aws_cdk import Aspects, CfnOutput, CfnResource, Duration, IAspect, RemovalPolicy, Stack
from aws_cdk import aws_amplify as amplify
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_athena as athena
from aws_cdk import aws_bedrock as bedrock
from aws_cdk import aws_verifiedpermissions as avp
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_glue as glue
from aws_cdk import aws_grafana as managed_grafana
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kinesis as kinesis
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_neptune as neptune
from aws_cdk import aws_opensearchserverless as aoss
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_scheduler as scheduler
from aws_cdk import aws_sqs as sqs
from aws_cdk import aws_stepfunctions as sfn
from constructs import Construct

HACKATHON_ROOT = str(Path(__file__).resolve().parents[3])   # Docker build context (contains researcher/, data/, maveric)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from atr import protocol  # noqa: E402  (pure: the ASL definition only)
from atr.checkpoint import load_policies  # noqa: E402  (pure: parses policies/atr.cedar)

# Event account rules: every service role sits under the /bb26/ path and carries the bb26 permissions boundary.
ROLE_PATH = "/bb26/"
BOUNDARY = "arn:aws:iam::{account}:policy/bb26-boundary/bb26-Boundary"


@jsii.implements(IAspect)
class EventRoleRules:
    """Applies the path and the boundary to every IAM role in the stack, including the ones CDK makes for its own custom
    resources (raw AWS::IAM::Role resources, which the L2 Role API does not reach)."""

    def __init__(self, boundary_arn: str):
        self.boundary = boundary_arn

    def visit(self, node):
        if isinstance(node, CfnResource) and node.cfn_resource_type == "AWS::IAM::Role":
            node.add_property_override("Path", ROLE_PATH)
            node.add_property_override("PermissionsBoundary", self.boundary)

FEATURE_TABLES = {
    "kpi_window": [("campaign", "string"), ("window", "int"), ("ts", "string"), ("cell", "int"), ("gnb", "string"),
                   ("ue_share", "double"), ("handovers_out", "int"), ("handovers_in", "int"), ("ping_pongs", "int"),
                   ("rlf", "int"), ("mean_rsrp_dbm", "double"), ("mean_sinr_db", "double"), ("prb_util_pct", "double")],
    "alarm": [("campaign", "string"), ("window", "int"), ("ts", "string"), ("cell", "int"), ("gnb", "string"),
              ("alarm", "string"), ("severity", "string"), ("text", "string")],
    "iface_event": [("campaign", "string"), ("window", "int"), ("ts", "string"), ("node", "string"), ("proto", "string"),
                    ("msg", "string"), ("category", "string"), ("ue", "int"), ("cell", "int"), ("cause", "string")],
}


def _table(scope, name):
    return ddb.Table(scope, name, partition_key=ddb.Attribute(name="id", type=ddb.AttributeType.STRING),
                     billing_mode=ddb.BillingMode.PAY_PER_REQUEST, removal_policy=RemovalPolicy.DESTROY)


def _bucket(scope, name):
    return s3.Bucket(scope, name, block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
                     removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True)


class ResearcherStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, phase: int = 1, grafana: bool = True, **kwargs):
        super().__init__(scope, construct_id, **kwargs)
        out = {}

        # ================================================================ PHASE 1: core
        Aspects.of(self).add(EventRoleRules(BOUNDARY.format(account=self.account)))
        bucket, lake = _bucket(self, "Field"), _bucket(self, "Lake")
        findings, audit, truth = _table(self, "Findings"), _table(self, "Audit"), _table(self, "Truth")
        # every decision and agent record carries its trace id: rebuild a trace with one query
        audit.add_global_secondary_index(index_name="trace", partition_key=ddb.Attribute(name="trace_id", type=ddb.AttributeType.STRING),
                                         sort_key=ddb.Attribute(name="epoch", type=ddb.AttributeType.NUMBER))
        registry = _table(self, "Registry")     # models and tools the checkpoint lets agents call; withdraw = pause

        # the checkpoint's policy decision point: policies/atr.cedar, one AVP policy per statement, named by its id
        store = avp.CfnPolicyStore(self, "Policies", description="Telecom Researcher checkpoint",
                                   validation_settings=avp.CfnPolicyStore.ValidationSettingsProperty(mode="OFF"))
        for p in load_policies():
            avp.CfnPolicy(self, f"Policy-{p.id}", policy_store_id=store.attr_policy_store_id,
                          definition=avp.CfnPolicy.PolicyDefinitionProperty(
                              static=avp.CfnPolicy.StaticPolicyDefinitionProperty(statement=p.text, description=p.id)))
        store_arn = f"arn:aws:verifiedpermissions::{self.account}:policy-store/{store.attr_policy_store_id}"

        # Amazon Bedrock Guardrails on every model input and output
        f = bedrock.CfnGuardrail.ContentFilterConfigProperty
        guard = bedrock.CfnGuardrail(
            self, "Guardrail", name="atr-agents", blocked_input_messaging="Blocked by the Telecom Researcher guardrail.",
            blocked_outputs_messaging="Blocked by the Telecom Researcher guardrail.",
            content_policy_config=bedrock.CfnGuardrail.ContentPolicyConfigProperty(filters_config=[
                *[f(type=t, input_strength="MEDIUM", output_strength="MEDIUM")
                  for t in ("HATE", "INSULTS", "SEXUAL", "VIOLENCE", "MISCONDUCT")],
                f(type="PROMPT_ATTACK", input_strength="HIGH", output_strength="NONE")]),
            sensitive_information_policy_config=bedrock.CfnGuardrail.SensitiveInformationPolicyConfigProperty(
                pii_entities_config=[bedrock.CfnGuardrail.PiiEntityConfigProperty(type=t, action="BLOCK")
                                     for t in ("AWS_ACCESS_KEY", "AWS_SECRET_KEY", "PASSWORD")]))
        guard_version = bedrock.CfnGuardrailVersion(self, "GuardrailV1", guardrail_identifier=guard.attr_guardrail_id)

        def image(cmd):
            return _lambda.DockerImageCode.from_image_asset(
                HACKATHON_ROOT, file="researcher/deploy/lambda_fn/Dockerfile", cmd=[cmd],
                exclude=[".venv", "researcher/runs", "researcher/deploy/cdk", "researcher/models/hf", "researcher/demo_video",
                         "**/__pycache__", "**/.git", "maveric-1.0.0 1/maveric-1.0.0/notebooks",
                         "maveric-1.0.0 1/maveric-1.0.0/docs", "data/mro_data", "*.pdf"])

        # services not deployed in this phase stay local inside the Lambdas (atr/modes.py toggle)
        env = {"ATR_MODE": "aws", "ATR_FIELD_BUCKET": bucket.bucket_name, "ATR_LAKE_BUCKET": lake.bucket_name,
               "ATR_DDB_TABLE": findings.table_name, "ATR_AUDIT_TABLE": audit.table_name, "ATR_TRUTH_TABLE": truth.table_name,
               "ATR_GATEWAY": "local", "ATR_LAB": "local", "ATR_PROTOCOL": "local", "ATR_MODEL_IMPORT": "local",
               "ATR_POLICY_STORE_ID": store.attr_policy_store_id, "ATR_REGISTRY_TABLE": registry.table_name,
               "ATR_EVIDENCE": "local"}
        if phase < 2:
            env.update(ATR_STREAM="local", ATR_FEATURES="local")
        if phase < 3:
            env.update(ATR_SEARCH="local", ATR_GRAPH="local")

        fn = _lambda.DockerImageFunction(self, "RunExperiment", memory_size=3008, timeout=Duration.seconds(120),
                                         code=image("handler.handler"), environment={"ATR_FIELD_BUCKET": bucket.bucket_name})
        bucket.grant_read_write(fn)
        lab_fn = _lambda.DockerImageFunction(self, "LabExecutor", memory_size=3008, timeout=Duration.minutes(5),
                                             code=image("lab_handler.handler"), environment=env)
        truth.grant_read_data(lab_fn)       # the lab holds the injected fault; the agents' role cannot read it

        gateway_role = iam.Role(self, "GatewayRole", assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"))
        fn.grant_invoke(gateway_role)
        lab_fn.grant_invoke(gateway_role)

        queue = sqs.Queue(self, "Jobs", visibility_timeout=Duration.minutes(15))
        events.Rule(self, "NightlySweep", schedule=events.Schedule.cron(minute="0", hour="2"),
                    targets=[targets.SqsQueue(queue, message=events.RuleTargetInput.from_object(
                        {"job": "overnight-sweep", "priority": 9}))])

        agent_role = iam.Role(self, "AgentRole", assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"))
        agent_role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"], resources=["*"]))
        agent_role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock-agentcore:InvokeCodeInterpreter", "bedrock-agentcore:StartCodeInterpreterSession",
                     "bedrock-agentcore:StopCodeInterpreterSession", "bedrock-agentcore:InvokeGateway"], resources=["*"]))
        findings.grant_read_write_data(agent_role)
        audit.grant_write_data(agent_role)
        queue.grant_consume_messages(agent_role)
        agent_role.add_to_policy(iam.PolicyStatement(      # AgentCore Runtime pulls the agents' image and writes logs/traces
            actions=["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:GetAuthorizationToken", "logs:CreateLogGroup",
                     "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams", "logs:DescribeLogGroups",
                     "xray:PutTraceSegments", "xray:PutTelemetryRecords", "cloudwatch:PutMetricData",
                     "bedrock-agentcore:GetWorkloadAccessToken*", "sqs:SendMessage"], resources=["*"]))
        bucket.grant_read(agent_role)
        agent_role.add_to_policy(iam.PolicyStatement(effect=iam.Effect.DENY, actions=["dynamodb:*"], resources=[truth.table_arn]))
        # the checkpoint: agents may ask for decisions and read the registry, never change policies or the registry
        agent_role.add_to_policy(iam.PolicyStatement(actions=["verifiedpermissions:IsAuthorized", "verifiedpermissions:GetPolicy"],
                                                     resources=[store_arn]))
        registry.grant_read_data(agent_role)
        agent_role.add_to_policy(iam.PolicyStatement(effect=iam.Effect.DENY, actions=["dynamodb:PutItem", "dynamodb:UpdateItem",
                                                     "dynamodb:DeleteItem", "dynamodb:BatchWriteItem"], resources=[registry.table_arn]))
        agent_role.add_to_policy(iam.PolicyStatement(actions=["bedrock:ApplyGuardrail"], resources=[guard.attr_guardrail_arn]))
        # the network language model (OTel-LLM-1.7B-IT on SageMaker, deploy/nlm_endpoint.py), called behind the checkpoint
        agent_role.add_to_policy(iam.PolicyStatement(actions=["sagemaker:InvokeEndpoint"],
                                                     resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/atr-otel-nlm"]))

        workers = [lab_fn]
        out.update(FieldBucket=bucket.bucket_name, LakeBucket=lake.bucket_name, FindingsTable=findings.table_name,
                   AuditTable=audit.table_name, TruthTable=truth.table_name, JobsQueueUrl=queue.queue_url,
                   RunExperimentLambdaArn=fn.function_arn, LabExecutorLambdaArn=lab_fn.function_arn,
                   GatewayRoleArn=gateway_role.role_arn, AgentRoleArn=agent_role.role_arn, Phase=str(phase),
                   PolicyStoreId=store.attr_policy_store_id, RegistryTable=registry.table_name,
                   GuardrailId=guard.attr_guardrail_id, GuardrailVersion=guard_version.attr_version)

        # ================================================================ PHASE 3 networking first (phase-2 functions join the VPC)
        vpc = fn_sg = graph = logs = None
        if phase >= 3:
            vpc = ec2.Vpc(self, "Vpc", max_azs=2, nat_gateways=1)
            graph_sg = ec2.SecurityGroup(self, "GraphSg", vpc=vpc, description="Neptune finding graph")
            fn_sg = ec2.SecurityGroup(self, "FnSg", vpc=vpc, description="researcher functions")
            graph_sg.add_ingress_rule(fn_sg, ec2.Port.tcp(8182), "openCypher from researcher functions")
            subnets = neptune.CfnDBSubnetGroup(self, "GraphSubnets", db_subnet_group_description="finding graph",
                                               subnet_ids=vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS).subnet_ids)
            graph = neptune.CfnDBCluster(self, "FindingGraph", iam_auth_enabled=True, engine_version="1.3.2.1",
                                         db_subnet_group_name=subnets.ref, vpc_security_group_ids=[graph_sg.security_group_id],
                                         serverless_scaling_configuration=neptune.CfnDBCluster.ServerlessScalingConfigurationProperty(
                                             min_capacity=1, max_capacity=8))
            neptune.CfnDBInstance(self, "FindingGraphInstance", db_instance_class="db.serverless", db_cluster_identifier=graph.ref)
            coll = "atr-logs"
            enc = aoss.CfnSecurityPolicy(self, "LogsEnc", name="atr-logs-enc", type="encryption", policy=json.dumps(
                {"Rules": [{"ResourceType": "collection", "Resource": [f"collection/{coll}"]}], "AWSOwnedKey": True}))
            net = aoss.CfnSecurityPolicy(self, "LogsNet", name="atr-logs-net", type="network", policy=json.dumps(
                [{"Rules": [{"ResourceType": "collection", "Resource": [f"collection/{coll}"]}], "AllowFromPublic": True}]))
            logs = aoss.CfnCollection(self, "Logs", name=coll, type="VECTORSEARCH")
            logs.node.add_dependency(enc)
            logs.node.add_dependency(net)
            env.update(ATR_OPENSEARCH_ENDPOINT=logs.attr_collection_endpoint, ATR_NEPTUNE_ENDPOINT=graph.attr_endpoint)
            # lab_fn was created before these endpoints existed: give it the search endpoint (it never reads the graph)
            lab_fn.add_environment("ATR_OPENSEARCH_ENDPOINT", logs.attr_collection_endpoint)
            lab_fn.add_environment("ATR_GRAPH", "local")
            out.update(OpenSearchEndpoint=logs.attr_collection_endpoint, NeptuneEndpoint=graph.attr_endpoint)

        # ================================================================ PHASE 2: full pipeline
        if phase >= 2:
            streams = {n: kinesis.Stream(self, f"Stream{n.title()}", stream_name=f"atr-{n}",
                                         stream_mode=kinesis.StreamMode.ON_DEMAND) for n in ("kpi", "alarms")}
            db = glue.CfnDatabase(self, "Features", catalog_id=self.account,
                                  database_input=glue.CfnDatabase.DatabaseInputProperty(name="atr_features"))
            for tname, cols in FEATURE_TABLES.items():
                t = glue.CfnTable(self, f"Table{tname.title().replace('_', '')}", catalog_id=self.account, database_name="atr_features",
                                  table_input=glue.CfnTable.TableInputProperty(
                                      name=tname, table_type="EXTERNAL_TABLE", parameters={"classification": "json"},
                                      storage_descriptor=glue.CfnTable.StorageDescriptorProperty(
                                          location=f"s3://{lake.bucket_name}/features/{tname}/",
                                          input_format="org.apache.hadoop.mapred.TextInputFormat",
                                          output_format="org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
                                          serde_info=glue.CfnTable.SerdeInfoProperty(serialization_library="org.openx.data.jsonserde.JsonSerDe"),
                                          columns=[glue.CfnTable.ColumnProperty(name=c, type=ty) for c, ty in cols])))
                t.node.add_dependency(db)
            wg = athena.CfnWorkGroup(self, "Athena", name="atr", work_group_configuration=athena.CfnWorkGroup.WorkGroupConfigurationProperty(
                result_configuration=athena.CfnWorkGroup.ResultConfigurationProperty(output_location=f"s3://{lake.bucket_name}/athena/")))
            env.update(ATR_KINESIS_PREFIX="atr", ATR_GLUE_DATABASE="atr_features", ATR_ATHENA_OUTPUT=f"s3://{lake.bucket_name}/athena/")
            for k in ("ATR_KINESIS_PREFIX", "ATR_GLUE_DATABASE", "ATR_ATHENA_OUTPUT"):
                lab_fn.add_environment(k, env[k])
            vpc_kw = {"vpc": vpc, "security_groups": [fn_sg]} if vpc else {}
            task_fn = _lambda.DockerImageFunction(self, "ProtocolTask", memory_size=4096, timeout=Duration.minutes(15),
                                                  code=image("protocol_handler.handler"), environment=env, **vpc_kw)
            api_fn = _lambda.DockerImageFunction(self, "ConsoleApi", memory_size=1024, timeout=Duration.seconds(30),
                                                 code=image("console_api.handler"), environment=env, **vpc_kw)
            truth.grant_read_data(task_fn)
            workers += [task_fn, api_fn]
            for st in streams.values():
                for f in workers:
                    st.grant_read_write(f)

            sm = sfn.CfnStateMachine(self, "Protocol", state_machine_name="atr-experiment-protocol",
                                     definition_string=json.dumps(protocol.definition()),
                                     definition_substitutions={"ProtocolTaskFunction": task_fn.function_arn},
                                     role_arn=self._sfn_role(task_fn).role_arn)
            sched_role = iam.Role(self, "SchedulerRole", assumed_by=iam.ServicePrincipal("scheduler.amazonaws.com"))
            sched_role.add_to_policy(iam.PolicyStatement(actions=["states:StartExecution"], resources=[sm.attr_arn]))
            scheduler.CfnSchedule(self, "NightlyProtocol", schedule_expression="cron(0 1 * * ? *)",
                                  flexible_time_window=scheduler.CfnSchedule.FlexibleTimeWindowProperty(mode="OFF"),
                                  target=scheduler.CfnSchedule.TargetProperty(arn=sm.attr_arn, role_arn=sched_role.role_arn,
                                                                              input=json.dumps({"run_id": "<aws.scheduler.execution-id>"})))
            for f in (task_fn, api_fn):
                queue.grant_send_messages(f)
                f.add_environment("ATR_SQS_URL", queue.queue_url)

            pool = cognito.UserPool(self, "Reviewers", self_sign_up_enabled=False, sign_in_aliases=cognito.SignInAliases(email=True),
                                    removal_policy=RemovalPolicy.DESTROY)
            domain = pool.add_domain("ReviewersDomain", cognito_domain=cognito.CognitoDomainOptions(domain_prefix=f"atr-{self.account}"))
            console_app = amplify.CfnApp(self, "Console", name="atr-review-console", platform="WEB")
            branch = amplify.CfnBranch(self, "ConsoleMain", app_id=console_app.attr_app_id, branch_name="main")
            console_url = f"https://main.{console_app.attr_default_domain}"
            client = pool.add_client("ConsoleClient", o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(implicit_code_grant=True), callback_urls=[console_url + "/"],
                scopes=[cognito.OAuthScope.OPENID, cognito.OAuthScope.EMAIL]))
            api = apigw.RestApi(self, "ConsoleRest", default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=[console_url], allow_headers=["Authorization", "Content-Type"]))
            auth = apigw.CognitoUserPoolsAuthorizer(self, "ConsoleAuth", cognito_user_pools=[pool])
            integ = apigw.LambdaIntegration(api_fn)
            r_api = api.root.add_resource("api")
            for path, method in (("findings", "GET"), ("graph", "GET"), ("verdict", "POST")):
                r_api.add_resource(path).add_method(method, integ, authorizer=auth, authorization_type=apigw.AuthorizationType.COGNITO)
            api_fn.add_environment("ATR_CONSOLE_ORIGIN", console_url)

            models = _bucket(self, "Models")
            sm_role = iam.Role(self, "TrainingRole", assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
                               managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")])
            models.grant_read_write(sm_role)
            import_role = iam.Role(self, "ModelImportRole", assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"))
            models.grant_read(import_role)
            out.update(StateMachineArn=sm.attr_arn, AthenaWorkGroup=wg.name, ConsoleUrl=console_url, ConsoleApiUrl=api.url,
                       CognitoDomain=domain.base_url(), CognitoClientId=client.user_pool_client_id, ConsoleBranch=branch.branch_name,
                       ConsoleAppId=console_app.attr_app_id,
                       ModelBucket=models.bucket_name, TrainingRoleArn=sm_role.role_arn, ModelImportRoleArn=import_role.role_arn)

        # ================================================================ PHASE 3: always-on services
        if phase >= 3:
            aoss.CfnAccessPolicy(self, "LogsAccess", name="atr-logs-access", type="data", policy=json.dumps([{
                "Rules": [{"ResourceType": "index", "Resource": ["index/atr-logs/*"], "Permission": ["aoss:*"]},
                          {"ResourceType": "collection", "Resource": ["collection/atr-logs"], "Permission": ["aoss:*"]}],
                "Principal": [f.role.role_arn for f in workers]}]))
        if phase >= 3 and grafana:      # Managed Grafana needs IAM Identity Center enabled in the account (buildspec detects it)
            grafana_role = iam.Role(self, "GrafanaRole", assumed_by=iam.ServicePrincipal("grafana.amazonaws.com"))
            grafana_role.add_to_policy(iam.PolicyStatement(actions=["athena:*", "glue:Get*", "s3:GetObject", "s3:ListBucket",
                                                                    "s3:PutObject", "s3:GetBucketLocation"], resources=["*"]))
            managed_grafana.CfnWorkspace(self, "Dashboards", account_access_type="CURRENT_ACCOUNT", authentication_providers=["AWS_SSO"],
                                 permission_type="SERVICE_MANAGED", data_sources=["ATHENA"], role_arn=grafana_role.role_arn,
                                 name="atr-telemetry")

        # permissions shared by every worker function
        for f in workers:
            for b in (bucket, lake):
                b.grant_read_write(f)
            findings.grant_read_write_data(f)
            audit.grant_write_data(f)
            registry.grant_read_data(f)
            f.add_to_role_policy(iam.PolicyStatement(actions=["verifiedpermissions:IsAuthorized", "verifiedpermissions:GetPolicy"],
                                                     resources=[store_arn]))
            f.add_to_role_policy(iam.PolicyStatement(actions=["bedrock:InvokeModel", "bedrock-agentcore:*"], resources=["*"]))
            if phase >= 2:
                f.add_to_role_policy(iam.PolicyStatement(actions=["athena:*", "glue:GetTable", "glue:GetDatabase", "glue:GetPartitions"],
                                                         resources=["*"]))
            if phase >= 3:
                f.add_to_role_policy(iam.PolicyStatement(actions=["aoss:APIAccessAll", "neptune-db:*"], resources=["*"]))

        for key, value in out.items():
            CfnOutput(self, key, value=value)

    def _sfn_role(self, task_fn):
        role = iam.Role(self, "ProtocolRole", assumed_by=iam.ServicePrincipal("states.amazonaws.com"))
        task_fn.grant_invoke(role)
        return role
