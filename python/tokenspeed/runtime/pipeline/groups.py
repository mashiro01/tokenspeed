"""Stable process-group role names for the native pipeline runtime."""

PIPELINE_RESULT_GROUP_ROLE = "pipeline-result"
PIPELINE_STEP_META_GROUP_ROLE = "pipeline-step-meta"
PIPELINE_FAULT_UPSTREAM_GROUP_ROLE = "pipeline-fault-upstream"
PIPELINE_FAULT_DOWNSTREAM_GROUP_ROLE = "pipeline-fault-downstream"

# A blocking Gloo recv inherits its process-group operation timeout. Keep the
# fault lane alive well inside that deadline so an idle, healthy service does
# not mistake "no fault reported" for a failed channel.
PIPELINE_FAULT_HEARTBEAT_SECONDS = 30.0
PIPELINE_FAULT_GROUP_TIMEOUT_SECONDS = 300
