// Telecom Researcher policies: the one checkpoint every model, tool and agent-to-agent call crosses.
// Evaluated by Amazon Verified Permissions in AWS mode (deploy/sync_policies.py uploads this file) and by
// atr/checkpoint.py LocalPdp offline. Each statement is one policy; the "// id:" line names it in decision records.
// Default is deny: a call no permit matches is refused. A forbid always wins over a permit.

// id: scientist-tools
permit (principal == Atr::Agent::"scientist",
        action in [Atr::Action::"observe_symptom", Atr::Action::"propose_claims", Atr::Action::"scan_network",
                   Atr::Action::"run_experiment"],
        resource)
when { resource.status == "approved" };

// id: experimenter-tools
permit (principal == Atr::Agent::"experimenter",
        action in [Atr::Action::"test_claim", Atr::Action::"run_control", Atr::Action::"run_null",
                   Atr::Action::"run_holdout", Atr::Action::"run_replication", Atr::Action::"test_insight",
                   Atr::Action::"replicate_insight", Atr::Action::"holdout_insight", Atr::Action::"run_experiment"],
        resource)
when { resource.status == "approved" };

// id: critic-tools
// The Critic judges evidence; it has no experiment tools, so it cannot manufacture its own evidence.
permit (principal == Atr::Agent::"critic",
        action in [Atr::Action::"review_claim", Atr::Action::"review_insight"],
        resource)
when { resource.status == "approved" };

// id: orchestrator-delegation
permit (principal == Atr::Agent::"orchestrator",
        action in [Atr::Action::"ask_scientist", Atr::Action::"ask_experimenter", Atr::Action::"ask_critic",
                   Atr::Action::"publish_findings", Atr::Action::"publish_insights"],
        resource)
when { resource.status == "approved" };

// id: pipeline-services
permit (principal == Atr::Agent::"protocol", action in [Atr::Action::"lab_execute"], resource)
when { resource.status == "approved" };

// id: observer-reads
permit (principal == Atr::Agent::"observer", action in [Atr::Action::"query_features", Atr::Action::"search_logs"], resource)
when { resource.status == "approved" };

// id: analyst-compute
permit (principal == Atr::Agent::"analyst", action in [Atr::Action::"effect_size"], resource)
when { resource.status == "approved" };

// id: agents-use-approved-models
permit (principal is Atr::Agent, action == Atr::Action::"InvokeModel", resource is Atr::Model)
when { resource.status == "approved" };

// id: withdrawn-is-refused
// The pause: withdrawing a model or a tool in the registry refuses every call to it, whatever else permits it.
forbid (principal, action, resource)
when { resource.status == "withdrawn" };

// id: humans-approve-changes
permit (principal is Atr::Human, action == Atr::Action::"ApproveChange", resource is Atr::Card);

// id: agents-never-change-state
// No agent may approve or apply a change of state; only a person can, through ApproveChange.
forbid (principal is Atr::Agent, action in [Atr::Action::"ApproveChange", Atr::Action::"ApplyNetworkChange"], resource);
