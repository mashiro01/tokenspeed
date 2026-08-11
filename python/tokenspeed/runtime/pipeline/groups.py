"""Stable process-group role names for the native pipeline runtime."""

PIPELINE_RESULT_GROUP_ROLE = "pipeline-result"
PIPELINE_P2P_GROUP_ROLE = "pipeline-p2p"
PIPELINE_STEP_META_GROUP_ROLE = "pipeline-step-meta"
PIPELINE_FAULT_UPSTREAM_GROUP_ROLE = "pipeline-fault-upstream"
PIPELINE_FAULT_DOWNSTREAM_GROUP_ROLE = "pipeline-fault-downstream"

# A blocking Gloo recv inherits its process-group operation timeout. Keep the
# fault lane alive well inside that deadline so an idle, healthy service does
# not mistake "no fault reported" for a failed channel.
PIPELINE_FAULT_HEARTBEAT_SECONDS = 30.0
PIPELINE_FAULT_GROUP_TIMEOUT_SECONDS = 300


def build_pipeline_p2p_groups(
    *, stage_count: int, stage_world_size: int
) -> tuple[tuple[int, int], ...]:
    """Return every adjacent lane pair plus the DSpark endpoint pair."""

    groups = [
        (
            lane + stage * stage_world_size,
            lane + (stage + 1) * stage_world_size,
        )
        for stage in range(stage_count - 1)
        for lane in range(stage_world_size)
    ]
    if stage_count > 2:
        groups.extend(
            (lane, lane + (stage_count - 1) * stage_world_size)
            for lane in range(stage_world_size)
        )
    return tuple(groups)
