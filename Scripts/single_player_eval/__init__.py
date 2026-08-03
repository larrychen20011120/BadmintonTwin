"""
Single Player Evaluation Module

這個模組提供針對單一選手的快速評估功能，
支援多種演算法：bc, rallynet, rallynetv2, naive_rag, pure_llm, random, timerag, twin, twin2

主要功能：
1. 模擬與Trajectory收集: eval_single_player.py, eval_all_rallies.py, evaluators/
2. Canonical trajectory scoring: score.py
3. Additional trajectory analysis and visualization: evaluate_trajectories.py

使用方法：
    # 模擬並收集 trajectory
    python Scripts/single_player_eval/eval_single_player.py \\
        --algorithm bc \\
        --player "Viktor AXELSEN" \\
        --save-trajectories

    # 計算 canonical metrics
    python Scripts/single_player_eval/score.py \\
        --run-root Experiments/bc/trajectories/rally_no_instruct \\
        --output-dir Experiments/bc/results/rally_no_instruct/latest
"""

# Evaluation tools (for analysis) - these can be imported independently
try:
    from .evaluate_trajectories import (
        evaluate_trajectories,
        SequenceMetrics,
        DistributionMetrics,
        ConditionalMetrics,
        Visualizer,
        load_trajectories,
        extract_sequences,
    )
    _evaluation_tools_available = True
except ImportError as e:
    _evaluation_tools_available = False
    _evaluation_error = str(e)

# Evaluators (for simulation/collection) - require full dependencies
try:
    from .evaluators.base_evaluator import BaseEvaluator
    from .evaluators.bc_evaluator import BCEvaluator
    from .evaluators.rallynet_evaluator import RallyNetEvaluator
    from .evaluators.rallynetv2_evaluator import RallyNetV2Evaluator
    from .evaluators.naive_rag_evaluator import NaiveRAGEvaluator
    from .evaluators.pure_llm_evaluator import PureLLMEvaluator
    from .evaluators.random_evaluator import RandomEvaluator
    from .evaluators.timerag_evaluator import TimeRAGEvaluator
    from .evaluators.twin_evaluator import TwinEvaluator
    _evaluators_available = True
except ImportError as e:
    _evaluators_available = False
    _evaluator_error = str(e)

__all__ = []

if _evaluation_tools_available:
    __all__.extend([
        'evaluate_trajectories',
        'SequenceMetrics',
        'DistributionMetrics',
        'ConditionalMetrics',
        'Visualizer',
        'load_trajectories',
        'extract_sequences',
    ])

if _evaluators_available:
    __all__.extend([
        'BaseEvaluator',
        'BCEvaluator',
        'RallyNetEvaluator',
        'RallyNetV2Evaluator',
        'NaiveRAGEvaluator',
        'PureLLMEvaluator',
        'RandomEvaluator',
        'TimeRAGEvaluator',
        'TwinEvaluator',
        'Twin2Evaluator',
    ])
