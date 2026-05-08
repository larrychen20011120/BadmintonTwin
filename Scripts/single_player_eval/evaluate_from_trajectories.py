"""
⚠️ DEPRECATED: This file has been replaced by evaluate_trajectories.py

Please use the new evaluation script instead:
    python Scripts/single_player_eval/evaluate_trajectories.py

The new script provides:
- More comprehensive metrics (DTW, CTC, Edit Distance)
- Better conditional distribution analysis
- Improved visualizations
- Cleaner CLI interface
"""

import warnings
warnings.warn(
    "This module is deprecated. Use evaluate_trajectories.py instead.",
    DeprecationWarning,
    stacklevel=2
)

# Keep old imports for backward compatibility
from Scripts.single_player_eval.evaluate_trajectories import (
    load_trajectories,
    extract_sequences,
    SequenceMetrics,
    DistributionMetrics,
    ConditionalMetrics,
    Visualizer,
    evaluate_trajectories,
)

__all__ = [
    'load_trajectories',
    'extract_sequences',
    'SequenceMetrics',
    'DistributionMetrics',
    'ConditionalMetrics',
    'Visualizer',
    'evaluate_trajectories',
]
