from __future__ import annotations

from src.config.settings import InferenceConfig
from src.core.inference import InferenceRunner
from src.stages.base import Stage, StageRegistry


@StageRegistry.register(
    name="inference",
    description="Load checkpoint and generate text; --leaderboard prints only output",
)
class InferenceStage(Stage):
    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()

    def run(self, **kwargs) -> None:
        runner = InferenceRunner(self.config)
        runner.run()
