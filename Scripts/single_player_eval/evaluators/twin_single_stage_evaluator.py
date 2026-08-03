"""Physical-rally evaluator for the single-stage Twin agent."""

from Agent.twin_single_stage import TwinSingleStageAgent

from .twin_evaluator import TwinEvaluator


class TwinSingleStageEvaluator(TwinEvaluator):
    """Evaluate Twin retrieval with one complete action generation."""

    algorithm_name = "twin_single_stage"
    agent_class = TwinSingleStageAgent
    retrieval_mode = "twin_aspect_single_stage_v1"
    uses_conditioned_retrieval = False
