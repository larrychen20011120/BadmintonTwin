"""
Single Player Evaluation Module

這個模組提供針對單一選手的快速評估功能，
支援多種演算法：bc, rallynet, rallynetv2, naive_rag, random, timerag, twin

主要功能：
1. 模擬與Trajectory收集: eval_single_player.py, evaluators/
2. Trajectory分析與評估: evaluate_trajectories.py

使用方法：
    # 模擬並收集 trajectory
    python Scripts/single_player_eval/eval_single_player.py \\
        --algorithm bc \\
        --player "Viktor AXELSEN" \\
        --save-trajectories

    # 分析 trajectory
    python Scripts/single_player_eval/evaluate_trajectories.py \\
        --trajectory-file Experiments/bc/trajectories/Viktor_AXELSEN_norm.pkl \\
        --metrics all \\
        --visualize
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
        'RandomEvaluator',
        'TimeRAGEvaluator',
        'TwinEvaluator',
    ])
