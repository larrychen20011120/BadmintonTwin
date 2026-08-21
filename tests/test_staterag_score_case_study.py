import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Scripts.single_player_eval.staterag_score_case_study import (
    SequentialScoreRunner,
    _score_path,
    build_service_state,
    run_condition_chain,
    summarize_score_run,
    write_score_latex,
)
from Utils.helper import action_col, state_col


class _DummyAgent:
    def reset_history(self):
        pass

    def predict(self, state, info, launch):
        shot_type = 1 if launch else 2
        return (shot_type, state[-1], [0.0, 0.0], [0.0, 0.0], [0.1] * 10), ""


class _DummyEvaluator:
    def __init__(self):
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


def _initial_state():
    row = {column: 0.0 for column in state_col + action_col}
    row.update({
        "rally": 1,
        "ball_round": 1,
        "player_score": 0,
        "opponent_score": 0,
        "score_status": 0,
        "player": "A",
        "opponent_type": 0,
    })
    return {
        column: str(row[column]) if column == "player" else row[column]
        for column in state_col
    }


def _spec(rallies=3, max_shots=2):
    return {
        "players": ["A", "B"],
        "target_player": "A",
        "opponent_player": "B",
        "initial_server": "A",
        "initial_score": {"A": 0, "B": 0},
        "rallies_per_condition": rallies,
        "max_shots": max_shots,
        "seed": 42,
        "source_data_point": {
            "key": [1, 1, 1],
            "initial_state": _initial_state(),
        },
        "conditions": [{
            "key": "no_instruction",
            "label": "No instruction",
            "instruction": None,
            "applied_instruction": None,
            "cluster_id": None,
        }],
    }


class _ScriptedRunner:
    def __init__(self, winners):
        self.winners = list(winners)
        self.calls = 0

    def simulate_scored_rally(self, condition, rally_index, rollout_seed, server, score):
        self.calls += 1
        winner = self.winners[rally_index - 1]
        loser = "B" if winner == "A" else "A"
        score_after = dict(score)
        score_after[winner] += 1
        return {
            "condition": condition["key"],
            "rally_index": rally_index,
            "rollout_index": rally_index,
            "server": server,
            "receiver": "B" if server == "A" else "A",
            "score_before": dict(score),
            "score_after": score_after,
            "winner": winner,
            "loser": loser,
            "next_server": winner,
            "target_player": "A",
            "opponent_player": "B",
            "target_player_won": winner == "A",
            "rally_length": 1,
            "termination_reason": "natural_terminate",
            "pred_sequence": [],
        }


class SequentialScoreCaseStudyTest(unittest.TestCase):
    def test_service_state_uses_server_score_perspective(self):
        state, receiver = build_service_state(
            _spec(),
            server="B",
            score={"A": 2, "B": 1},
            rally_index=4,
            player_ids={"A": 0, "B": 1},
        )
        self.assertEqual(receiver, "A")
        self.assertEqual(int(state[0].item()), 4)
        self.assertEqual(int(state[1].item()), 1)
        self.assertEqual(int(state[2].item()), 1)
        self.assertEqual(int(state[3].item()), 2)
        self.assertEqual(int(state[4].item()), -1)
        self.assertEqual(int(state[17].item()), 1)

    def test_cap_scores_point_and_makes_winner_next_server(self):
        spec = _spec(rallies=1, max_shots=2)
        runner = SequentialScoreRunner.__new__(SequentialScoreRunner)
        runner.normal = _DummyEvaluator()
        runner.instructed = _DummyEvaluator()
        runner.spec = spec
        trajectory = runner.simulate_scored_rally(
            spec["conditions"][0],
            rally_index=1,
            rollout_seed=7,
            server="A",
            score={"A": 0, "B": 0},
        )
        self.assertEqual(trajectory["termination_reason"], "max_shots_reached")
        self.assertEqual(trajectory["winner"], "A")
        self.assertEqual(trajectory["score_after"], {"A": 1, "B": 0})
        self.assertEqual(trajectory["next_server"], "A")

    def test_chain_updates_score_and_resumes_contiguous_prefix(self):
        spec = _spec()
        condition = spec["conditions"][0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            runner = _ScriptedRunner(["A", "B", "B"])
            result = run_condition_chain(spec, condition, run_dir, runner=runner)
            self.assertEqual(result["final_score"], {"A": 1, "B": 2})
            self.assertEqual(runner.calls, 3)
            second = pd.read_json(_score_path(run_dir, condition["key"], 2), typ="series")
            self.assertEqual(second["server"], "A")
            self.assertEqual(second["score_before"], {"A": 1, "B": 0})

            resumed_runner = _ScriptedRunner(["A", "B", "B"])
            resumed = run_condition_chain(
                spec,
                condition,
                run_dir,
                runner=resumed_runner,
            )
            self.assertEqual(resumed["final_score"], {"A": 1, "B": 2})
            self.assertEqual(resumed_runner.calls, 0)

    def test_summary_and_latex_report_score_progression(self):
        spec = _spec()
        condition = spec["conditions"][0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            run_condition_chain(
                spec,
                condition,
                run_dir,
                runner=_ScriptedRunner(["A", "B", "A"]),
            )
            summary = summarize_score_run(spec, run_dir)
            latex_path = write_score_latex(spec, summary, run_dir)
            item = summary["conditions"][condition["key"]]
            self.assertEqual(item["final_score"], {"A": 2, "B": 1})
            self.assertEqual(len(item["score_progression"]), 3)
            report = latex_path.read_text(encoding="utf-8")
            self.assertIn("Sequential-Score StateRAG Case Study", report)
            self.assertIn("Score progression", report)


if __name__ == "__main__":
    unittest.main()
