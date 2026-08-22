from datetime import datetime, timezone

from app.schemas.agent_evaluation import (
    AgentEvaluationObservation,
    AgentEvaluationObservationSet,
)


def test_observation_set_accepts_toolset_release_metadata():
    observations = AgentEvaluationObservationSet(
        dataset_id="dataset",
        dataset_version="1.0.0",
        runner_version="runner",
        toolset_version="toolset-v2:test",
        tool_names=["search_knowledge", "mcp__release_probe__echo"],
        generated_at=datetime.now(timezone.utc),
        observations=[
            AgentEvaluationObservation(
                case_id="case-1",
                run_succeeded=True,
                latency_ms=1,
            )
        ],
    )

    assert observations.toolset_version == "toolset-v2:test"
    assert "mcp__release_probe__echo" in observations.tool_names
