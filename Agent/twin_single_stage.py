import json
import os
import time

from jinja2 import Template

from Agent.base_agent import BaseAgent
from Agent.twin import (
    DEFAULT_EMBEDDING_MODEL,
    TwinAgent,
    encode_texts,
    project_path,
    safe_float,
    safe_int,
)
from Utils.langfuse_tracing import BadmintonTracer


class TwinSingleStageAgent(TwinAgent):
    """Aspect-conditioned Twin retrieval with one complete action generation."""

    def __init__(self, player_name, database, embedding_cache, model_name=None, api_key=None,
                 host=None, port=None, base_url=None, host_url=None, normal_top_k=10,
                 miss_top_k=5, lambda_weight=0.5,
                 embedding_model_name=DEFAULT_EMBEDDING_MODEL,
                 embedding_batch_size=None, embedding_revision=None,
                 normalize_embeddings=True, embedding_dtype="float32",
                 embedding_model=None, enable_tracing=True,
                 instruction_mode="no_instruct", embedding_provider="local",
                  embedding_dimension=None, conditioned_top_k=None,
                  case_grounding=True, save_prompt_traces=False,
                  semantic_start_ball_round=3,
                  online_terminal_aspect_repair=True):
        del conditioned_top_k
        super().__init__(
            player_name=player_name,
            database=database,
            embedding_cache=embedding_cache,
            model_name=model_name,
            api_key=api_key,
            host=host,
            port=port,
            base_url=base_url,
            host_url=host_url,
            normal_top_k=normal_top_k,
            miss_top_k=miss_top_k,
            conditioned_top_k=1,
            lambda_weight=lambda_weight,
            embedding_model_name=embedding_model_name,
            embedding_batch_size=embedding_batch_size,
            embedding_revision=embedding_revision,
            normalize_embeddings=normalize_embeddings,
            embedding_dtype=embedding_dtype,
            embedding_model=embedding_model,
            enable_tracing=enable_tracing,
            instruction_mode=instruction_mode,
            embedding_provider=embedding_provider,
            embedding_dimension=embedding_dimension,
            case_grounding=case_grounding,
            save_prompt_traces=save_prompt_traces,
            semantic_start_ball_round=semantic_start_ball_round,
            online_terminal_aspect_repair=online_terminal_aspect_repair,
        )
        self.retrieval_architecture = "single_stage_v1"
        self.tracer = BadmintonTracer(
            "twin_single_stage",
            player_name,
            enable_tracing,
            capture_local=save_prompt_traces,
        )
        prompt_dir = os.path.join(project_path, "Prompts", "New", "twin_single_stage")
        if self.case_grounding:
            prompt_name = f"action_{instruction_mode}.jinja"
        else:
            prompt_name = (
                "action_direct_instruct.jinja"
                if instruction_mode == "instruct" else "action_direct.jinja"
            )
        with open(os.path.join(prompt_dir, prompt_name), "r") as handle:
            self.action_template = Template(handle.read())
        if self.case_grounding:
            self.action_schema_name = "twin_single_stage_grounded"
            schema_path = os.path.join(
                project_path,
                "Schemas",
                "Agent",
                "twin_single_stage_grounded_schema.json",
            )
            with open(schema_path, "r", encoding="utf-8") as handle:
                self.action_response_schema = json.load(handle)
        else:
            self.action_response_schema, self.action_schema_name = (
                BaseAgent.get_rag_single_stage_response_schema()
            )

    @staticmethod
    def _parse_case_assessments(response, case_grounding, normal_cases, miss_cases):
        if not case_grounding:
            return []
        normal_cases = list(normal_cases or [])
        miss_cases = list(miss_cases or [])
        expected_identities = [
            ("miss", rank) for rank in range(1, len(miss_cases) + 1)
        ] + [
            ("success", rank) for rank in range(1, len(normal_cases) + 1)
        ]
        return BaseAgent._parse_rag_case_assessments(
            response.get("case_assessments"),
            BaseAgent.RAG_STAGE1_ASSESSMENT_KEYS,
            expected_identities,
            sources={"success", "miss"},
        )

    @classmethod
    def _parse_action_response(cls, response, launch, available_types,
                               case_grounding=True, normal_cases=None,
                               miss_cases=None):
        if not case_grounding:
            action = BaseAgent.parse_rag_single_stage_response(
                response,
                launch=launch,
                available_types=available_types,
            )
            return action, ""
        expected_keys = (
            {"case_assessments", "action"}
            if case_grounding else {"action"}
        )
        if not isinstance(response, dict) or set(response) != expected_keys:
            return None, ""
        case_assessments = cls._parse_case_assessments(
            response, case_grounding, normal_cases, miss_cases
        )
        if case_assessments is None:
            return None, ""
        action, _ = BaseAgent.parse_prompt_action(response)
        if action is None:
            return None, ""
        shot_type = int(action["shot_type"])
        if shot_type == 11:
            return action, ""
        legal_types = {1, 10} if launch else set(range(2, 10))
        if shot_type not in legal_types or shot_type not in set(available_types):
            return None, ""
        return action, ""

    def _fallback_case_assessments(self, retrieval):
        if not self.case_grounding:
            return []
        return [
            {"source": "miss", "retrieval_rank": rank,
             "impact": "irrelevant", "factors": ["assessment unavailable"]}
            for rank in range(1, len(retrieval["miss_cases"]) + 1)
        ] + [
            {"source": "success", "retrieval_rank": rank,
             "impact": "irrelevant", "factors": ["assessment unavailable"]}
            for rank in range(1, len(retrieval["normal_cases"]) + 1)
        ]

    def _fallback_action(self, retrieval, launch):
        nearest = self._nearest_case(retrieval["normal_cases"] + retrieval["miss_cases"])
        if nearest is None:
            return {
                "outcome": "returned",
                "shot_type": 1 if launch else 2,
                "landing_x": 0.0,
                "landing_y": 0.0,
                "moving_x": 0.0,
                "moving_y": 0.0,
            }, "fallback_default"

        shot_type = safe_int(nearest.get("player_type"))
        if shot_type == 11:
            coordinates = {
                "landing_x": None,
                "landing_y": None,
                "moving_x": None,
                "moving_y": None,
            }
        else:
            coordinates = {
                key: safe_float(nearest.get(key))
                for key in ("landing_x", "landing_y", "moving_x", "moving_y")
            }
        return {
            "outcome": "missed" if shot_type == 11 else "returned",
            "shot_type": shot_type,
            **coordinates,
        }, "fallback_nearest_case"

    def _format_memories_for_prompt(self, memories, include_semantics=True):
        return [
            {
                "decision_description": self.verbalizer.build_case_verbalization(
                    memory
                )["stage1_description"],
                "context_text": (
                    str(memory.get("semantic_text", "")).strip()
                    if include_semantics else ""
                ),
                "reachability_text": memory.get("reachability_text", ""),
                "action_verbalization": memory.get("action_verbalization", ""),
            }
            for memory in memories
        ]

    def predict(self, state, info, launch):
        state_dict = self._build_state_dict_from_env(state, info)
        rally_num = info.get("rally", 0)
        if self._current_rally != rally_num:
            self._current_rally = rally_num
            self.trajectory = []
        shared_history = info.get("rally_history")
        state_history = (
            self._normalize_history_for_player(shared_history)
            if shared_history is not None else list(self.trajectory)
        )
        self.trajectory.append(state_dict)

        self.tracer.start_trace(state_dict, info, launch)
        profile_instruction = info.get("rally_profile_instruction", "")
        semantic_enabled = self._semantic_enabled(state_dict)
        aspect_repair = None
        if semantic_enabled:
            aspects, aspect_source, aspect_repair = self._estimate_aspects(
                state_dict,
                state_history,
            )
            query_embedding = encode_texts(
                self.embedding_model,
                [aspects["semantic_query_text"]],
                self.embedding_dim,
                self.embedding_batch_size,
                normalize_embeddings=self.normalize_embeddings,
                input_type="query",
                model_name=self.embedding_model_name,
            )[0]
            retrieval_mode = "state_semantic"
        else:
            aspects = {}
            aspect_source = "skipped_early_round"
            query_embedding = None
            retrieval_mode = "state_only"

        started_at = time.time()
        if semantic_enabled:
            retrieval = self.database.retrieve_dual(
                state_dict,
                query_embedding,
                normal_top_k=self.normal_top_k,
                miss_top_k=self.miss_top_k,
                lambda_weight=self.lambda_weight,
                launch=launch,
            )
        else:
            retrieval = self.database.retrieve_state_only_stage1(
                state_dict,
                normal_top_k=self.normal_top_k,
                miss_top_k=self.miss_top_k,
                launch=launch,
            )
        memories = retrieval["normal_cases"] + retrieval["miss_cases"]
        self.tracer.log_retrieval(
            state_dict,
            memories,
            "twin",
            int((time.time() - started_at) * 1000),
            stage="single_stage",
        )

        context = self.verbalizer.build_prompt_context_nl(
            state_dict, [], launch=launch, state_history=None
        )
        available_types = self.database.available_shot_types(launch=launch)
        normal_memories = self._format_memories_for_prompt(
            retrieval["normal_cases"], include_semantics=semantic_enabled
        )
        miss_memories = self._format_memories_for_prompt(
            retrieval["miss_cases"], include_semantics=semantic_enabled
        )
        user_prompt = self.action_template.render(
            trajectory_text=context["trajectory_text"],
            semantic_enabled=semantic_enabled,
            semantic_query_text=aspects.get("semantic_query_text", ""),
            normal_memories=normal_memories,
            miss_memories=miss_memories,
            rally_profile_instruction=profile_instruction,
            case_grounding=self.case_grounding,
        )
        action_schema = (
            BaseAgent.constrain_case_assessment_count(
                self.action_response_schema,
                len(retrieval["normal_cases"]) + len(retrieval["miss_cases"]),
            )
            if self.case_grounding else self.action_response_schema
        )
        started_at = time.time()
        response, raw_llm_output, usage = BaseAgent.get_verified_response_or_none(
            self.llm,
            schema=action_schema,
            schema_name=self.action_schema_name,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_trials=2,
            validator=lambda candidate: self._parse_action_response(
                candidate,
                launch,
                available_types,
                self.case_grounding,
                retrieval["normal_cases"],
                retrieval["miss_cases"],
            )[0] is not None,
        )
        self.tracer.log_generation(
            self.system_prompt,
            user_prompt,
            response,
            latency_ms=int((time.time() - started_at) * 1000),
            raw_llm_output=raw_llm_output,
            usage=usage,
            stage="action",
        )
        parsed_action, _ = self._parse_action_response(
            response,
            launch,
            available_types,
            self.case_grounding,
            retrieval["normal_cases"],
            retrieval["miss_cases"],
        )
        if parsed_action is None:
            parsed_action, action_source = self._fallback_action(retrieval, launch)
            case_assessments = self._fallback_case_assessments(retrieval)
        else:
            action_source = "llm"
            case_assessments = self._parse_case_assessments(
                response,
                self.case_grounding,
                retrieval["normal_cases"],
                retrieval["miss_cases"],
            )

        shot_type = int(parsed_action["shot_type"])
        if shot_type == 11:
            landing_x = landing_y = moving_x = moving_y = 0.0
        else:
            landing_x = float(parsed_action["landing_x"])
            landing_y = float(parsed_action["landing_y"])
            moving_x = float(parsed_action["moving_x"])
            moving_y = float(parsed_action["moving_y"])

        if self.trajectory:
            self.trajectory[-1].update({
                "player_type": shot_type,
                "landing_x": landing_x,
                "landing_y": landing_y,
                "moving_x": moving_x,
                "moving_y": moving_y,
            })
        action = (
            shot_type,
            state[-1] if len(state) > 0 else None,
            [landing_x, landing_y],
            [moving_x, moving_y],
            [0.1] * 10,
        )
        reasoning = ""
        self.tracer.end_trace(
            {
                "shot_type": shot_type,
                "landing": [landing_x, landing_y],
                "movement": [moving_x, moving_y],
            },
            reasoning,
        )

        normal_refs = self._case_references(retrieval["normal_cases"], 1, "normal")
        miss_refs = self._case_references(retrieval["miss_cases"], 1, "miss")
        assessed_refs = self._map_case_assessments(
            case_assessments,
            {"normal": normal_refs, "miss": miss_refs},
            stage=1,
        )
        all_case_ids = normal_refs + miss_refs
        self._last_case_ids = all_case_ids
        pre_stage = {
            "architecture": self.pre_stage_architecture,
            "semantic_enabled": semantic_enabled,
            "activation_ball_round": self.semantic_start_ball_round,
            "source": aspect_source,
        }
        if semantic_enabled:
            pre_stage.update({
                "semantic_query_text": aspects["semantic_query_text"],
                "aspects": {
                    key: aspects.get(key, "")
                    for key in (
                        "my_tactical_pattern",
                        "my_tempo_movement_dynamics",
                        "opponent_tactical_pattern",
                        "opponent_tempo_movement_dynamics",
                    )
                },
            })
            if aspect_repair is not None:
                pre_stage["repair"] = aspect_repair
        source_prefix = aspect_source if semantic_enabled else "state_only"
        self.last_decision_trace = {
            "schema_version": 7,
            "architecture": self.retrieval_architecture,
            "case_grounding": self.case_grounding,
            "outcome": "missed" if shot_type == 11 else "returned",
            "retrieval_mode": retrieval_mode,
            "pre_stage": pre_stage,
            "retrieval": {
                "normal_case_ids": normal_refs,
                "miss_case_ids": miss_refs,
            },
            "case_ids": all_case_ids,
            "case_assessments": assessed_refs,
            "action": {
                "shot_type": shot_type,
                "landing_x": None if shot_type == 11 else landing_x,
                "landing_y": None if shot_type == 11 else landing_y,
                "moving_x": None if shot_type == 11 else moving_x,
                "moving_y": None if shot_type == 11 else moving_y,
            },
            "action_source": f"{source_prefix}+{action_source}",
        }
        return action, reasoning
