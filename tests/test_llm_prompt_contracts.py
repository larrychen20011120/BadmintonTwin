from pathlib import Path
import unittest

from jinja2 import Environment, StrictUndefined

from Agent.naive_rag import NaiveRAGVerbalizer
from Agent.pure_llm import PureLLMAgent
from Agent.timerag import TimeRAGVerbalizer
from Scripts.single_player_eval.evaluators.base_evaluator import resolve_prompt_variant
from Utils.badminton_config import EN_SHOT2INDEX, shot_type_to_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = PROJECT_ROOT / "Prompts" / "New"
ONE_STAGE_TEMPLATE_PATHS = (
    PROMPT_ROOT / "pure_llm" / "user.jinja",
)
RAG_TEMPLATE_PATHS = tuple(
    PROMPT_ROOT / algorithm / f"stage{stage}_{variant}.jinja"
    for algorithm in ("naive_rag", "timerag")
    for stage in (1, 2)
    for variant in ("norm", "instruct")
)
TWIN_RAG_TEMPLATE_PATHS = tuple(
    PROMPT_ROOT / "twin" / f"stage{stage}_{variant}.jinja"
    for stage in (1, 2)
    for variant in ("no_instruct", "instruct")
)


def render_strict(path, **context):
    environment = Environment(undefined=StrictUndefined)
    return environment.from_string(path.read_text()).render(**context)


class LLMPromptStructureTest(unittest.TestCase):
    def test_prompt_variant_follows_instruction_mode(self):
        expected = {
            ("norm", "auto"): "norm",
            ("norm", "off"): "norm",
            ("shifted", "auto"): "shifted",
            ("shifted", "off"): "norm",
        }
        for parameters, variant in expected.items():
            self.assertEqual(resolve_prompt_variant(*parameters), variant)
        with self.assertRaises(ValueError):
            resolve_prompt_variant("shifted", "invalid")

    def test_prompts_are_grouped_by_algorithm(self):
        expected = set(ONE_STAGE_TEMPLATE_PATHS) | set(RAG_TEMPLATE_PATHS) | set(TWIN_RAG_TEMPLATE_PATHS) | {
            PROMPT_ROOT / "twin" / "aspect_user.jinja",
            PROMPT_ROOT / "twin" / "memory_user.jinja",
        }
        self.assertTrue(all(path.is_file() for path in expected))
        self.assertEqual(list(PROMPT_ROOT.glob("*.jinja")), [])

    def test_system_prompt_defines_shared_court_and_action_rules(self):
        content = (PROMPT_ROOT / "simulate_system.md").read_text()
        for text in (
            "Court Area Definitions",
            "Coordinate System",
            "Valid grid coordinates",
            "semantic area must match",
            "Missed shot",
            "Serve short",
            "Smash Defence",
        ):
            self.assertIn(text, content)

    def test_one_stage_templates_define_shared_contract(self):
        for path in ONE_STAGE_TEMPLATE_PATHS:
            with self.subTest(path=path):
                content = path.read_text()
                self.assertIn("imitation task", content)
                self.assertIn("not tactical optimization", content)
                self.assertIn("Missed shot", content)
                self.assertIn("legal return into the opponent's court", content)
                self.assertTrue(
                    "Jointly analyze" in content
                    or "jointly analyzing" in content
                    or "Make one joint decision" in content
                )

    def test_rag_agents_use_two_stage_templates(self):
        for algorithm in ("naive_rag", "timerag", "twin"):
            source = (PROJECT_ROOT / "Agent" / f"{algorithm}.py").read_text()
            for text in ("stage1_template", "stage2_template", "retrieve_stage1", "retrieve_by_shot_type"):
                self.assertIn(text, source)
            for path in RAG_TEMPLATE_PATHS:
                if path.parent.name == algorithm:
                    self.assertIn("imitation task", path.read_text())

    def test_two_stage_output_contracts_are_disjoint(self):
        for algorithm in ("naive_rag", "timerag"):
            stage1 = (PROMPT_ROOT / algorithm / "stage1_norm.jinja").read_text()
            stage2 = (PROMPT_ROOT / algorithm / "stage2_norm.jinja").read_text()
            self.assertIn('"outcome": "returned"', stage1)
            self.assertNotIn('"landing_x"', stage1)
            self.assertIn('"placement"', stage2)
            self.assertIn("Do not include shot_type", stage2)
            self.assertNotIn("selected_shot_type", stage2)
        stage1 = (PROMPT_ROOT / "twin" / "stage1_no_instruct.jinja").read_text()
        stage2 = (PROMPT_ROOT / "twin" / "stage2_no_instruct.jinja").read_text()
        self.assertIn('"outcome": "returned"', stage1)
        self.assertNotIn('"landing_x"', stage1)
        self.assertIn('"placement"', stage2)
        self.assertIn("Do not include shot_type", stage2)

    def test_rag_prompts_use_stage_specific_evidence_without_distance(self):
        for path in RAG_TEMPLATE_PATHS + TWIN_RAG_TEMPLATE_PATHS:
            with self.subTest(path=path):
                content = path.read_text()
                expected_field = (
                    "case.stage1_description" if "stage1_" in path.name
                    else "case.stage2_description"
                )
                self.assertIn(expected_field, content)
                self.assertNotIn("case.description", content)
                self.assertNotIn("case.distance", content)
                self.assertNotIn("distance", content.lower())
                self.assertIn("evidence-driven behavior simulation", content)
                self.assertIn("not tactical optimization", content)
                self.assertIn("movement", content)
                self.assertIn("score pressure", content)

    def test_stage1_prompts_hardcode_serve_and_rally_shot_types(self):
        rally_types = (
            "Clear, Push Shot, Smash, Smash Defence, Drive, Net Shot, Lob, or Drop"
        )
        for algorithm in ("naive_rag", "timerag"):
            for variant in ("norm", "instruct"):
                content = (
                    PROMPT_ROOT / algorithm / f"stage1_{variant}.jinja"
                ).read_text()
                self.assertIn("Serve short or Serve long", content)
                self.assertIn(rally_types, content)
                self.assertNotIn("available_shot_types", content)
        for variant in ("no_instruct", "instruct"):
            content = (PROMPT_ROOT / "twin" / f"stage1_{variant}.jinja").read_text()
            self.assertIn("Serve short or Serve long", content)
            self.assertIn(rally_types, content)

    def test_instruct_prompts_apply_weak_tactical_instruction(self):
        for algorithm in ("naive_rag", "timerag"):
            stage1 = (PROMPT_ROOT / algorithm / "stage1_instruct.jinja").read_text()
            stage2 = (PROMPT_ROOT / algorithm / "stage2_instruct.jinja").read_text()
            for content in (stage1, stage2):
                self.assertIn("## Tactical Imitation Instruction", content)
                self.assertIn("weak auxiliary information", content)
                self.assertIn("Keep the prediction case-driven", content)
                self.assertIn("otherwise similarly relevant", content)
                self.assertNotIn("Player Profile Evidence", content)
            self.assertNotIn(
                "rally_profile_instruction",
                (PROMPT_ROOT / algorithm / "stage1_norm.jinja").read_text(),
            )
            self.assertNotIn(
                "rally_profile_instruction",
                (PROMPT_ROOT / algorithm / "stage2_norm.jinja").read_text(),
            )
        for stage in (1, 2):
            content = (PROMPT_ROOT / "twin" / f"stage{stage}_instruct.jinja").read_text()
            self.assertIn("## Tactical Imitation Instruction", content)
            self.assertIn("weak auxiliary information", content)
            self.assertIn("Keep the prediction case-driven", content)
            self.assertIn("otherwise similarly relevant", content)
            self.assertNotIn(
                "rally_profile_instruction",
                (PROMPT_ROOT / "twin" / f"stage{stage}_no_instruct.jinja").read_text(),
            )

    def test_twin_uses_aspect_pre_stage_and_two_stage_action_templates(self):
        source = (PROJECT_ROOT / "Agent" / "twin.py").read_text()
        self.assertIn("aspect_template", source)
        self.assertIn("stage1_template", source)
        self.assertIn("stage2_template", source)

    def test_twin_memory_prompt_uses_compact_phase_segments(self):
        content = (PROMPT_ROOT / "twin" / "memory_user.jinja").read_text()
        self.assertIn(
            '"range": [{{ first_ball_round }}, {{ last_ball_round }}]',
            content,
        )
        self.assertIn('"phase":', content)
        for removed_field in (
            '"phase_name":',
            '"narrative":',
            '"transition_reason":',
            '"evidence_strokes":',
        ):
            self.assertNotIn(removed_field, content)

    def test_twin_aspect_prompt_does_not_request_semantic_query(self):
        content = (PROMPT_ROOT / "twin" / "aspect_user.jinja").read_text()
        self.assertIn("current dominant phase", content)
        self.assertNotIn('"semantic_query_text":', content)

    def test_twin_phase_prompts_detect_sustained_turning_points(self):
        memory = (PROMPT_ROOT / "twin" / "memory_user.jinja").read_text()
        aspect = (PROMPT_ROOT / "twin" / "aspect_user.jinja").read_text()
        self.assertIn("Actively identify meaningful turning points", memory)
        self.assertIn("Do not merge clearly different phases", memory)
        self.assertIn('"range": [1, 6]', memory)
        self.assertIn('"range": [7, 12]', memory)
        self.assertIn("phase active at the latest stroke", aspect)
        self.assertIn("Do not average an earlier phase", aspect)

    def test_retrieval_summary_is_not_part_of_prompt_contract(self):
        for path in PROMPT_ROOT.rglob("*"):
            if path.is_file():
                self.assertNotIn("retrieval_summary", path.read_text())

    def test_agents_reference_algorithm_prompt_directories(self):
        expected = {
            "Agent/pure_llm.py": 'os.path.join(prompt_root, "pure_llm")',
            "Agent/naive_rag.py": 'os.path.join(prompt_root, "naive_rag")',
            "Agent/timerag.py": 'os.path.join(prompt_root, "timerag")',
            "Agent/twin.py": 'os.path.join(prompt_root, "twin")',
        }
        for relative_path, reference in expected.items():
            source = (PROJECT_ROOT / relative_path).read_text()
            self.assertIn(reference, source)
            if relative_path == "Agent/pure_llm.py":
                self.assertNotIn("simulate_system.md", source)
                self.assertNotIn("self.system_prompt", source)
            else:
                self.assertIn("simulate_system.md", source)


class LLMPromptRenderingTest(unittest.TestCase):
    def setUp(self):
        self.case = {
            "index": 1,
            "stage2_description": "A complete retrieved example.",
        }
        self.normal_case = {
            "index": 1,
            "stage1_description": "A normal retrieved example.",
        }
        self.miss_case = {
            "index": 1,
            "stage1_description": "A missed-shot retrieved example.",
        }

    def _stage1_context(self):
        return {
            "trajectory_text": "Complete current rally.",
            "normal_cases": [self.normal_case],
            "miss_cases": [self.miss_case],
            "rally_profile_instruction": "Front-Court Pressure. Prefer front-court pressure.",
            "semantic_query_text": "Twin semantic query.",
        }

    def _stage2_context(self):
        return {
            "trajectory_text": "Complete current rally.",
            "selected_shot_type": "Clear",
            "conditioned_cases": [self.case],
            "rally_profile_instruction": "Front-Court Pressure. Prefer front-court pressure.",
            "semantic_query_text": "Twin semantic query.",
        }

    def test_pure_llm_templates_render_strictly(self):
        template = PROMPT_ROOT / "pure_llm" / "user.jinja"
        rendered = render_strict(template, trajectory_text="Complete current rally.")
        self.assertIn("Complete current rally.", rendered)
        self.assertNotIn("style_shift_instruction", template.read_text())
        self.assertNotIn("rally_profile_instruction", template.read_text())
        self.assertFalse((PROMPT_ROOT / "pure_llm" / "shifted_user.jinja").exists())

    def test_pure_llm_normalizes_shared_history_from_each_player_view(self):
        agent = PureLLMAgent.__new__(PureLLMAgent)
        agent.player_name = "A"
        history = [{"player": "A"}, {"player": "B"}]
        normalized = agent._normalize_history_for_player(history)
        self.assertEqual(
            [(item["is_target_turn"], item["turn_role"]) for item in normalized],
            [(True, "target"), (False, "opponent")],
        )

    def test_rag_templates_render_strictly(self):
        for algorithm in ("naive_rag", "timerag"):
            for variant in ("norm", "instruct"):
                stage1 = render_strict(
                    PROMPT_ROOT / algorithm / f"stage1_{variant}.jinja",
                    **self._stage1_context(),
                )
                self.assertIn("A normal retrieved example.", stage1)
                self.assertIn("A missed-shot retrieved example.", stage1)
                stage2 = render_strict(
                    PROMPT_ROOT / algorithm / f"stage2_{variant}.jinja",
                    **self._stage2_context(),
                )
                self.assertIn("A complete retrieved example.", stage2)
                self.assertEqual(
                    "Front-Court Pressure. Prefer front-court pressure." in stage2,
                    variant == "instruct",
                )
        for variant in ("no_instruct", "instruct"):
            stage1 = render_strict(
                PROMPT_ROOT / "twin" / f"stage1_{variant}.jinja",
                **self._stage1_context(),
            )
            stage2 = render_strict(
                PROMPT_ROOT / "twin" / f"stage2_{variant}.jinja",
                **self._stage2_context(),
            )
            self.assertIn("Twin semantic query.", stage1)
            self.assertIn("A complete retrieved example.", stage2)
            self.assertEqual(
                "Front-Court Pressure. Prefer front-court pressure." in stage2,
                variant == "instruct",
            )


class MissedShotContractTest(unittest.TestCase):
    def test_canonical_missed_shot_name(self):
        self.assertEqual(EN_SHOT2INDEX["Missed shot"], 11)
        self.assertEqual(shot_type_to_index("Missed shot"), 11)
        with self.assertRaises(ValueError):
            shot_type_to_index("Miss shots")

    def test_stage1_verbalizers_keep_evidence_separate(self):
        retrieval = {"normal_cases": [], "miss_cases": []}
        state = {"ball_round": 2, "opponent_type": 2}
        naive = NaiveRAGVerbalizer().build_dual_prompt_context_nl(state, retrieval)
        timerag = TimeRAGVerbalizer().build_dual_prompt_context_nl(state, retrieval)
        self.assertIn("normal_cases", naive)
        self.assertIn("miss_cases", timerag)


if __name__ == "__main__":
    unittest.main()
