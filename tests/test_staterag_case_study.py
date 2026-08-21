import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Scripts.single_player_eval.staterag_case_study import (
    CLUSTER_FEATURE_COLUMNS,
    StateRAGCaseRunner,
    _atomic_write_json,
    _latex_escape,
    _trajectory_path,
    select_diverse_instructions,
    summarize_run,
    write_latex_report,
)
from Utils.helper import action_col, state_col


def _cluster(cluster_id, train_size, first_coordinate):
    center = {name: 0.0 for name in CLUSTER_FEATURE_COLUMNS}
    center[CLUSTER_FEATURE_COLUMNS[0]] = float(first_coordinate)
    return {
        "cluster_id": cluster_id,
        "train_size": train_size,
        "label": f"Profile {cluster_id}",
        "instruction": f"Instruction {cluster_id}",
        "center_z": center,
    }


class _DummyAgent:
    def reset_history(self):
        pass

    def predict(self, state, info, launch):
        shot_type = 1 if launch else 2
        return (shot_type, state[-1], [0.0, 0.0], [0.0, 0.0], [0.1] * 10), ""


class _DummyEvaluator:
    def __init__(self, dataframe):
        self.df = dataframe
        self.player2index = {"A": 0, "B": 1}
        self.agents = {"A": _DummyAgent(), "B": _DummyAgent()}

    def _get_agent(self, player):
        return self.agents[player]

    @staticmethod
    def _state_to_env_format(state):
        return [
            int(state[12].item()),
            (state[5].item(), state[6].item()),
            (state[13].item(), state[14].item()),
            (state[10].item(), state[11].item()),
        ]

    @staticmethod
    def build_rollout_info(match_id, rally_number, state, last_action):
        return {"match_id": match_id, "rally": rally_number, "action": [last_action]}

    @staticmethod
    def _build_history_stroke(state, player, action):
        return {"player": player, "ball_round": int(state[1].item())}


def _model_dataframe():
    row = {column: 0.0 for column in state_col + action_col}
    row.update({
        "match_id": 1,
        "set": 1,
        "rally": 1,
        "ball_round": 1,
        "player": "A",
        "player_type": 1,
        "opponent_type": 0,
    })
    return pd.DataFrame([row])


def _statistics():
    return {
        "smoothing_baselines": {
            "attack_shot_ratio": 0.0,
            "defense_shot_ratio": 0.0,
            "side_ratio": 0.0,
            "frontcourt_ratio": 0.0,
            "backcourt_ratio": 0.0,
            "front_back_switch_ratio": 0.0,
            "left_right_switch_ratio": 0.0,
            "signed_side_bias": 0.0,
        },
        "imputation_medians": {name: 0.0 for name in CLUSTER_FEATURE_COLUMNS},
        "means": {name: 0.0 for name in CLUSTER_FEATURE_COLUMNS},
        "stds": {name: 1.0 for name in CLUSTER_FEATURE_COLUMNS},
        "clipping": {
            name: {"lower": -10.0, "upper": 10.0}
            for name in ("log_rally_length", "average_movement", "landing_opponent_distance")
        },
    }


def _trajectory(condition, rollout_index, target_won, landing_x):
    target = "A"
    opponent = "B"
    winner, loser = (target, opponent) if target_won else (opponent, target)
    common_state = {
        "player_location_x": 0.0,
        "player_location_y": -0.5,
        "ball_distance": 0.5,
        "ball_distance_x": 0.0,
        "ball_distance_y": 0.5,
        "hit_x": 0.0,
        "hit_y": 0.0,
        "opponent_type": 2,
        "opponent_location_x": 0.0,
        "opponent_location_y": -0.5,
        "opponent_move_x": 0.0,
        "opponent_move_y": 0.0,
    }
    return {
        "schema_version": 1,
        "algorithm": "staterag (naive_rag)",
        "condition": condition,
        "rollout_index": rollout_index,
        "match_id": 1,
        "set": 1,
        "rally": 1,
        "players": [target, opponent],
        "target_player": target,
        "opponent_player": opponent,
        "pred_sequence": [
            {
                "player": target,
                "ball_round": 2,
                "shot_type": 4,
                "landing_x": landing_x,
                "landing_y": 0.5,
                "moving_x": 0.0,
                "moving_y": -0.5,
                **common_state,
            },
            {
                "player": opponent,
                "ball_round": 3,
                "shot_type": 11,
                "landing_x": None,
                "landing_y": None,
                "moving_x": None,
                "moving_y": None,
                **common_state,
            },
        ],
        "rally_length": 2,
        "termination_reason": "natural_terminate",
        "winner": winner,
        "loser": loser,
        "target_player_won": target_won,
    }


class StateRAGCaseStudyTest(unittest.TestCase):
    def test_selects_largest_then_farthest_instruction_clusters(self):
        clusters = [
            _cluster(0, 100, 0.0),
            _cluster(1, 20, 1.0),
            _cluster(2, 10, 5.0),
        ]
        selected = select_diverse_instructions(clusters, count=2)
        self.assertEqual([cluster["cluster_id"] for cluster in selected], [0, 2])

    def test_cap_assigns_loss_to_actor_of_last_shot(self):
        runner = StateRAGCaseRunner.__new__(StateRAGCaseRunner)
        runner.normal = _DummyEvaluator(_model_dataframe())
        runner.instructed = _DummyEvaluator(_model_dataframe())
        runner.spec = {
            "record": {
                "match_id": 1,
                "set": 1,
                "rally": 1,
                "players": ["A", "B"],
            },
            "target_player": "A",
            "opponent_player": "B",
            "max_shots": 2,
            "source_data_point": {"initial_state": {}},
            "cap_rule": "cap",
            "natural_end_rule": "natural",
        }
        trajectory = runner.simulate(
            {"key": "no_instruction", "instruction": None},
            rollout_index=1,
            rollout_seed=7,
        )
        self.assertEqual(trajectory["termination_reason"], "max_shots_reached")
        self.assertEqual(trajectory["loser"], "B")
        self.assertEqual(trajectory["winner"], "A")
        self.assertEqual(trajectory["point_score"], {"A": 1, "B": 0})
        self.assertTrue(trajectory["target_player_won"])

    def test_summary_and_latex_include_baseline_and_representatives(self):
        baseline = {
            "key": "no_instruction",
            "label": "No instruction",
            "cluster_id": None,
            "instruction": None,
            "center_z": None,
        }
        instructed = {
            "key": "instruction_01_cluster_1",
            "label": "Attack & net",
            "cluster_id": 1,
            "instruction": "Attack & finish.",
            "applied_instruction": "Attack & net. Attack & finish.",
            "center_z": {name: 0.0 for name in CLUSTER_FEATURE_COLUMNS},
        }
        spec = {
            "target_player": "A",
            "opponent_player": "B",
            "source_data_point": {"key": [1, 1, 1]},
            "rollouts_per_condition": 2,
            "max_shots": 24,
            "conditions": [baseline, instructed],
            "target_train_statistics": _statistics(),
            "smoothing_alpha": 2.0,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            for condition in spec["conditions"]:
                for rollout_index in (1, 2):
                    trajectory = _trajectory(
                        condition["key"],
                        rollout_index,
                        target_won=rollout_index == 1,
                        landing_x=-0.5 if rollout_index == 1 else 0.5,
                    )
                    _atomic_write_json(
                        _trajectory_path(run_dir, condition["key"], rollout_index),
                        trajectory,
                    )

            summary = summarize_run(spec, run_dir)
            latex_path = write_latex_report(spec, summary, run_dir)

            self.assertEqual(summary["conditions"]["no_instruction"]["target_win_rate"], 0.5)
            self.assertIn(
                "distance_improvement_over_no_instruction",
                summary["conditions"][instructed["key"]]["instruction_adherence"],
            )
            report = latex_path.read_text(encoding="utf-8")
            self.assertIn("StateRAG Instruction Case Study", report)
            self.assertIn("Attack \\& net", report)
            self.assertIn("tikzpicture", report)

    def test_latex_escape_handles_table_characters(self):
        self.assertEqual(_latex_escape("A_B & 50%"), r"A\_B \& 50\%")


if __name__ == "__main__":
    unittest.main()
