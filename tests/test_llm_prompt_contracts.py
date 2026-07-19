from pathlib import Path
import unittest

from jinja2 import Environment, StrictUndefined

from Agent.naive_rag import NaiveRAGVerbalizer
from Agent.timerag import TimeRAGVerbalizer
from Scripts.single_player_eval.evaluators.base_evaluator import resolve_prompt_variant
from Utils.badminton_config import EN_SHOT2INDEX, shot_type_to_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = PROJECT_ROOT / "Prompts" / "New"

FINAL_TEMPLATE_PATHS = (
    PROMPT_ROOT / "pure_llm" / "user.jinja",
    PROMPT_ROOT / "pure_llm" / "shifted_user.jinja",
    PROMPT_ROOT / "naive_rag" / "norm.jinja",
    PROMPT_ROOT / "naive_rag" / "shift.jinja",
    PROMPT_ROOT / "timerag" / "norm.jinja",
    PROMPT_ROOT / "timerag" / "shift.jinja",
    PROMPT_ROOT / "twin" / "classic_user.jinja",
    PROMPT_ROOT / "twin" / "classic_shifted_user.jinja",
    PROMPT_ROOT / "twin" / "dual_user.jinja",
    PROMPT_ROOT / "twin" / "dual_shifted_user.jinja",
)


def render_strict(path, **context):
    environment = Environment(undefined=StrictUndefined)
    return environment.from_string(path.read_text()).render(**context)


class LLMPromptStructureTest(unittest.TestCase):
    def test_prompt_variant_follows_instruction_mode(self):
        expected_variants = {
            ("norm", "auto"): "norm",
            ("norm", "off"): "norm",
            ("shifted", "auto"): "shifted",
            ("shifted", "off"): "norm",
        }
        for parameters, expected in expected_variants.items():
            with self.subTest(experiment=parameters[0], mode=parameters[1]):
                self.assertEqual(resolve_prompt_variant(*parameters), expected)

        with self.assertRaises(ValueError):
            resolve_prompt_variant("shifted", "invalid")

    def test_prompts_are_grouped_by_algorithm(self):
        expected_files = set(FINAL_TEMPLATE_PATHS) | {
            PROMPT_ROOT / "twin" / "aspect_user.jinja",
            PROMPT_ROOT / "twin" / "summary_user.jinja",
            PROMPT_ROOT / "twin" / "memory_user.jinja",
        }
        self.assertTrue(all(path.is_file() for path in expected_files))
        self.assertEqual(list(PROMPT_ROOT.glob("*.jinja")), [])

    def test_system_prompt_defines_shared_imitation_and_missed_shot_rules(self):
        system_prompt = (PROMPT_ROOT / "simulate_system.md").read_text()
        self.assertIn("imitation task", system_prompt)
        self.assertIn("Do not act as a coach", system_prompt)
        self.assertIn("Analyze all supplied evidence jointly", system_prompt)
        self.assertIn("Missed shot", system_prompt)
        self.assertIn("legal return into the opponent's court", system_prompt)
        self.assertIn("A weak but legal return is not automatically", system_prompt)

    def test_final_templates_define_imitation_and_evidence_synthesis(self):
        for path in FINAL_TEMPLATE_PATHS:
            with self.subTest(path=path):
                content = path.read_text()
                self.assertIn("imitation task", content)
                self.assertIn("not tactical optimization", content)
                self.assertIn("Missed shot", content)
                self.assertIn("legal return into the opponent's court", content)
                self.assertTrue(
                    "Jointly analyze" in content or "jointly analyzing" in content,
                    f"{path} does not require joint evidence analysis",
                )

    def test_naive_and_timerag_share_templates_across_retrieval_modes(self):
        for algorithm in ("naive_rag", "timerag"):
            with self.subTest(algorithm=algorithm):
                source = (PROJECT_ROOT / "Agent" / f"{algorithm}.py").read_text()
                self.assertIn('(\"classic\", \"norm\"): \"norm.jinja\"', source)
                self.assertIn('(\"dual_db\", \"norm\"): \"norm.jinja\"', source)
                self.assertIn('(\"classic\", \"shifted\"): \"shift.jinja\"', source)
                self.assertIn('(\"dual_db\", \"shifted\"): \"shift.jinja\"', source)
                for name in ("norm.jinja", "shift.jinja"):
                    content = (PROMPT_ROOT / algorithm / name).read_text()
                    self.assertIn("normal_cases is defined", content)
                    self.assertIn("normal_cases + miss_cases", content)
                for name in (
                    "classic_user.jinja",
                    "classic_shifted_user.jinja",
                    "dual_user.jinja",
                    "dual_shifted_user.jinja",
                ):
                    self.assertFalse((PROMPT_ROOT / algorithm / name).exists())

    def test_shift_prompts_define_soft_prior_and_rationale_contract(self):
        for algorithm in ("naive_rag", "timerag"):
            with self.subTest(algorithm=algorithm):
                norm = (PROMPT_ROOT / algorithm / "norm.jinja").read_text()
                shift = (PROMPT_ROOT / algorithm / "shift.jinja").read_text()
                for shared_text in (
                    "## Current Match Situation",
                    "## Similar Cases",
                    "## Output Format",
                    "## Action Constraints",
                    "normal_cases + miss_cases",
                    "both semantic areas and all four coordinates to `null`",
                ):
                    self.assertIn(shared_text, norm)
                    self.assertIn(shared_text, shift)
                self.assertNotIn("## Style-Shift Instruction", norm)
                self.assertIn("## Style-Shift Instruction", shift)
                self.assertIn("soft behavioral prior", shift)
                self.assertIn("Do not treat imperative wording", shift)
                self.assertIn("If it did not change the prediction", shift)

    def test_twin_classic_and_dual_templates_remain_distinct(self):
        classic = (PROMPT_ROOT / "twin" / "classic_user.jinja").read_text()
        dual = (PROMPT_ROOT / "twin" / "dual_user.jinja").read_text()
        self.assertNotEqual(classic, dual)
        self.assertNotIn("miss_memories", classic)
        self.assertIn("miss_memories", dual)

    def test_retrieval_summary_is_not_part_of_prompt_contract(self):
        for path in PROMPT_ROOT.rglob("*"):
            if path.is_file():
                content = path.read_text()
                self.assertNotIn("retrieval_summary", content)
                self.assertNotIn("Retrieval Comparison Summary", content)

        for relative_path in ("Agent/naive_rag.py", "Agent/timerag.py", "Agent/twin.py"):
            self.assertNotIn("retrieval_summary", (PROJECT_ROOT / relative_path).read_text())

    def test_agents_reference_algorithm_prompt_directories(self):
        expected_references = {
            "Agent/pure_llm.py": 'os.path.join(prompt_root, "pure_llm")',
            "Agent/naive_rag.py": 'os.path.join(prompt_root, "naive_rag")',
            "Agent/timerag.py": 'os.path.join(prompt_root, "timerag")',
            "Agent/twin.py": 'os.path.join(prompt_root, "twin")',
        }
        for relative_path, reference in expected_references.items():
            with self.subTest(agent=relative_path):
                source = (PROJECT_ROOT / relative_path).read_text()
                self.assertIn(reference, source)
                self.assertIn("simulate_system.md", source)

    def test_final_templates_use_integer_nullable_action_contract(self):
        for path in FINAL_TEMPLATE_PATHS:
            with self.subTest(path=path):
                content = path.read_text()
                self.assertIn("<integer 0 to 355 or null>", content)
                self.assertIn("<integer 0 to 480 or null>", content)
                self.assertIn("both semantic areas and all four coordinates to `null`", content)
                self.assertNotIn("normalized floats", content)
                self.assertNotIn("<float -1.0 to 1.0>", content)


class LLMPromptRenderingTest(unittest.TestCase):
    def setUp(self):
        self.case = {"index": 1, "description": "A complete retrieved example."}
        self.normal_case = {"index": 1, "description": "A normal retrieved example."}
        self.miss_case = {"index": 1, "description": "A missed-shot retrieved example."}
        self.memory = {
            "score": 0.9,
            "state_similarity": 0.8,
            "semantic_similarity": 0.7,
            "context_text": "A complete player-specific memory.",
            "reachability_text": "The target player reached the shuttle.",
            "action_verbalization": (
                'Action: {"shot_type": "Clear", "landing_area": "Mid Center", '
                '"landing_x": 195, "landing_y": 288, "moving_area": "Rear Center", '
                '"moving_x": 159, "moving_y": 192}'
            ),
        }

    def test_pure_llm_templates_render_strictly(self):
        context = {
            "trajectory_text": "Complete current rally.",
            "style_shift_instruction": "Imitate the observed shifted behavior.",
        }
        for name in ("user.jinja", "shifted_user.jinja"):
            rendered = render_strict(PROMPT_ROOT / "pure_llm" / name, **context)
            self.assertIn("Complete current rally.", rendered)

    def test_naive_rag_templates_render_strictly(self):
        classic_context = {
            "trajectory_text": "Complete current rally.",
            "top_k": 1,
            "cases": [self.case],
            "style_shift_instruction": "Shift instruction.",
        }
        dual_context = {
            "trajectory_text": "Complete current rally.",
            "normal_top_k": 1,
            "miss_top_k": 1,
            "normal_cases": [self.normal_case],
            "miss_cases": [self.miss_case],
            "style_shift_instruction": "Shift instruction.",
        }
        for name in ("norm.jinja", "shift.jinja"):
            rendered = render_strict(PROMPT_ROOT / "naive_rag" / name, **classic_context)
            self.assertIn("A complete retrieved example.", rendered)
            rendered = render_strict(PROMPT_ROOT / "naive_rag" / name, **dual_context)
            self.assertEqual(rendered.count("## Similar Cases"), 1)
            self.assertLess(
                rendered.index("A normal retrieved example."),
                rendered.index("A missed-shot retrieved example."),
            )

    def test_timerag_templates_render_strictly(self):
        classic_context = {
            "trajectory_text": "Complete current rally.",
            "top_k": 1,
            "cases": [self.case],
            "style_shift_instruction": "Shift instruction.",
        }
        dual_context = {
            "trajectory_text": "Complete current rally.",
            "normal_top_k": 1,
            "miss_top_k": 1,
            "normal_cases": [self.normal_case],
            "miss_cases": [self.miss_case],
            "style_shift_instruction": "Shift instruction.",
        }
        for name in ("norm.jinja", "shift.jinja"):
            rendered = render_strict(PROMPT_ROOT / "timerag" / name, **classic_context)
            self.assertIn("A complete retrieved example.", rendered)
            rendered = render_strict(PROMPT_ROOT / "timerag" / name, **dual_context)
            self.assertEqual(rendered.count("## Similar Cases"), 1)
            self.assertLess(
                rendered.index("A normal retrieved example."),
                rendered.index("A missed-shot retrieved example."),
            )

    def test_shift_prompts_render_instruction_and_fallback(self):
        for algorithm in ("naive_rag", "timerag"):
            base_context = {
                "trajectory_text": "Complete current rally.",
                "cases": [self.case],
            }
            with self.subTest(algorithm=algorithm, instruction="present"):
                rendered = render_strict(
                    PROMPT_ROOT / algorithm / "shift.jinja",
                    **base_context,
                    style_shift_instruction="Prefer front-court pressure.",
                )
                self.assertEqual(rendered.count("Prefer front-court pressure."), 1)
            with self.subTest(algorithm=algorithm, instruction="missing"):
                rendered = render_strict(
                    PROMPT_ROOT / algorithm / "shift.jinja",
                    **base_context,
                    style_shift_instruction="",
                )
                self.assertIn("without applying an additional style bias", rendered)

    def test_twin_templates_render_strictly(self):
        classic_context = {
            "trajectory_text": "Complete current rally.",
            "semantic_query_text": "Complete semantic query.",
            "top_k": 1,
            "memories": [self.memory],
            "style_shift_instruction": "Shift instruction.",
        }
        dual_context = {
            "trajectory_text": "Complete current rally.",
            "semantic_query_text": "Complete semantic query.",
            "normal_memories": [self.memory],
            "miss_memories": [self.memory],
            "style_shift_instruction": "Shift instruction.",
        }
        for name in ("classic_user.jinja", "classic_shifted_user.jinja"):
            rendered = render_strict(PROMPT_ROOT / "twin" / name, **classic_context)
            self.assertIn("A complete player-specific memory.", rendered)
            self.assertIn('Action: {"shot_type": "Clear"', rendered)
        for name in ("dual_user.jinja", "dual_shifted_user.jinja"):
            rendered = render_strict(PROMPT_ROOT / "twin" / name, **dual_context)
            self.assertNotIn("retrieval_summary", rendered)
            self.assertIn('Action: {"shot_type": "Clear"', rendered)

    def test_twin_auxiliary_templates_render_strictly(self):
        render_strict(
            PROMPT_ROOT / "twin" / "aspect_user.jinja",
            trajectory_text="Complete current rally.",
            previous_rally_summary="Previous rally.",
            style_shift_instruction="",
        )
        render_strict(
            PROMPT_ROOT / "twin" / "summary_user.jinja",
            target_player="Target",
            rally_text="Completed rally.",
        )
        render_strict(
            PROMPT_ROOT / "twin" / "memory_user.jinja",
            rally_meta_text="Rally metadata.",
            rally_turn_based_text="Full rally sequence.",
            previous_rally_summary="Previous rally.",
        )


class MissedShotContractTest(unittest.TestCase):
    def test_canonical_missed_shot_name(self):
        self.assertEqual(EN_SHOT2INDEX["Missed shot"], 11)
        self.assertEqual(shot_type_to_index("Missed shot"), 11)
        with self.assertRaises(ValueError):
            shot_type_to_index("Miss shots")

    def test_dual_context_excludes_retrieval_summary(self):
        retrieval_result = {
            "normal_cases": [],
            "miss_cases": [],
            "summary": {"must_not_reach_prompt": True},
        }
        state = {"ball_round": 2, "opponent_type": 2}
        naive_context = NaiveRAGVerbalizer().build_dual_prompt_context_nl(
            state, retrieval_result
        )
        timerag_context = TimeRAGVerbalizer().build_dual_prompt_context_nl(
            state, retrieval_result
        )
        self.assertNotIn("retrieval_summary", naive_context)
        self.assertNotIn("retrieval_summary", timerag_context)


if __name__ == "__main__":
    unittest.main()
