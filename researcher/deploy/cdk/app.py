import os

import aws_cdk as cdk

from stack import ResearcherStack

app = cdk.App()
phase = int(app.node.try_get_context("phase") or 1)      # cdk deploy -c phase=1|2|3 (see stack.py)
grafana = str(app.node.try_get_context("grafana") or "true").lower() == "true"
ResearcherStack(app, "TelecomResearcher", phase=phase, grafana=grafana,
                env=cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region=os.environ.get("CDK_DEFAULT_REGION")))
app.synth()
