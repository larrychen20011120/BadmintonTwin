import tempfile
import unittest
from pathlib import Path

from Scripts.single_player_eval.render_staterag_case_study import (
    adherence_marks,
    rally_actions,
    render_trajectory_comparison,
    target_actions,
)
from Scripts.single_player_eval.staterag_case_study import _atomic_write_json


def _shot(player, ball_round, shot_type, landing):
    return {
        "player": player,
        "ball_round": ball_round,
        "shot_type": shot_type,
        "landing_x": landing[0],
        "landing_y": landing[1],
        "moving_x": 0.0,
        "moving_y": 0.0,
    }


def _trajectory(condition, shots):
    return {
        "condition": condition,
        "rally_index": 1,
        "target_player": "A",
        "opponent_player": "B",
        "server": "A",
        "score_before": {"A": 2, "B": 0},
        "pred_sequence": shots,
        "rally_length": max(shot["ball_round"] for shot in shots),
        "termination_reason": "natural_terminate",
        "winner": "A",
    }


class RenderStateRAGCaseStudyTest(unittest.TestCase):
    def test_target_actions_use_player_and_global_numbering(self):
        trajectory = _trajectory("condition", [
            _shot("B", 1, 1, (0.0, 0.8)),
            _shot("A", 2, 8, (0.0, 0.0)),
            _shot("B", 3, 2, (0.0, 0.0)),
            _shot("A", 4, 7, (0.0, 0.8)),
        ])
        actions = target_actions(trajectory)
        self.assertEqual(
            [(action["player_action"], action["global_round"]) for action in actions],
            [(1, 2), (2, 4)],
        )

    def test_adherence_marks_strict_behavior(self):
        actions = [
            {
                "shot_type": "Lob",
                "landing_area": "Rear Center",
            },
            {
                "shot_type": "Drive",
                "landing_area": "Rear Right Side",
            },
            {
                "shot_type": "Drive",
                "landing_area": "Front Right Side",
            },
        ]
        defensive, _ = adherence_marks(actions, "defensive")
        front, _ = adherence_marks(actions, "front")
        drive, pairs = adherence_marks(actions, "drive_sequence")
        self.assertEqual(defensive, {0})
        self.assertEqual(front, {2})
        self.assertEqual(drive, {1, 2})
        self.assertEqual(pairs, [(1, 2)])

    def test_rally_actions_include_both_players_and_renumber_excerpt(self):
        trajectory = _trajectory("condition", [
            _shot("B", 5, 3, (0.0, 0.0)),
            _shot("A", 6, 6, (0.0, 0.0)),
            _shot("B", 7, 6, (0.0, 0.8)),
        ])
        actions = rally_actions(trajectory, first_round=5, last_round=7)
        self.assertEqual([action["shot"] for action in actions], [1, 2, 3])
        self.assertEqual([action["player"] for action in actions], ["B", "A", "B"])

    def test_renders_three_side_by_side_complete_rally_tables(self):
        conditions = [
            {
                "key": "no_instruction",
                "instruction": None,
                "applied_instruction": None,
            },
            {
                "key": "instruction_01_cluster_0",
                "applied_instruction": "Use deep defensive resets.",
            },
            {
                "key": "instruction_02_cluster_1",
                "applied_instruction": "Play to the front court.",
            },
            {
                "key": "instruction_03_cluster_2",
                "applied_instruction": "Drive rear then front.",
            },
        ]
        trajectories = {
            ("no_instruction", 10): _trajectory(
                "no_instruction",
                [
                    _shot("A" if round_number % 2 else "B", round_number, 2, (0.0, 0.0))
                    for round_number in range(1, 15)
                ],
            ),
            ("instruction_01_cluster_0", 3): _trajectory(
                "instruction_01_cluster_0",
                [
                    _shot("A", 1, 1, (0.0, 0.8)),
                    _shot("B", 2, 3, (0.0, 0.0)),
                    _shot("A", 3, 2, (0.0, 0.0)),
                ],
            ),
            ("instruction_02_cluster_1", 12): _trajectory(
                "instruction_02_cluster_1",
                [
                    _shot("A", 1, 1, (0.0, 0.8)),
                    _shot("B", 2, 8, (0.0, 0.0)),
                    _shot("A", 3, 7, (0.0, 0.8)),
                ],
            ),
            ("instruction_03_cluster_2", 2): _trajectory(
                "instruction_03_cluster_2",
                [
                    _shot("B", 5, 3, (0.0, 0.0)),
                    _shot("A", 6, 6, (0.0, 0.0)),
                    _shot("B", 7, 6, (0.0, 0.8)),
                    _shot("A", 8, 7, (0.0, 0.8)),
                ],
            ),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            _atomic_write_json(run_dir / "manifest.json", {
                "target_player": "A",
                "conditions": conditions,
            })
            for (condition, rally), trajectory in trajectories.items():
                _atomic_write_json(
                    run_dir / "conditions" / condition / f"rally_{rally:03d}.json",
                    trajectory,
                )

            output_path = render_trajectory_comparison(run_dir)
            rendered = output_path.read_text(encoding="utf-8")

        self.assertEqual(rendered.count(r"\begin{table*}[t]"), 1)
        self.assertEqual(rendered.count(r"\begin{tabularx}{\textwidth}"), 1)
        self.assertIn("Normal", rendered)
        self.assertIn(r"\textcolor{defensiveblue}", rendered)
        self.assertIn(r"\textcolor{frontgreen}", rendered)
        self.assertIn(r"\textcolor{driveorange}", rendered)
        self.assertIn(r"\mbox{\textbf{1.}", rendered)
        self.assertIn("Player", rendered)
        self.assertIn("Opponent", rendered)
        self.assertIn(r"\ldots", rendered)
        self.assertIn(r"\footnotesize", rendered)
        self.assertIn(r"\raggedright", rendered)
        self.assertIn("Use deep defensive resets.", rendered)
        self.assertIn("Player serves; score 2--0.", rendered)
        self.assertIn("wins (", rendered)
        self.assertNotIn("Instruction.", rendered)
        self.assertNotIn("Shot 1 &", rendered)
        self.assertNotIn("terminal error", rendered)
        self.assertNotIn("terminalred", rendered)
        self.assertNotIn("tikzpicture", rendered)


if __name__ == "__main__":
    unittest.main()
