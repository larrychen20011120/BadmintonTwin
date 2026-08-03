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
                 embedding_dimension=None, conditioned_top_k=None):
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
        )
        self.retrieval_architecture = "single_stage_v1"
        self.tracer = BadmintonTracer("twin_single_stage", player_name, enable_tracing)
        prompt_dir = os.path.join(project_path, "Prompts", "New", "twin_single_stage")
        with open(os.path.join(prompt_dir, f"action_{instruction_mode}.jinja"), "r") as handle:
            self.action_template = Template(handle.read())

    @staticmethod
    def _parse_action_response(response, launch, available_types):
        action, reasoning = BaseAgent.parse_prompt_action(response)
        if action is None or not isinstance(reasoning, str):
            return None, reasoning
        shot_type = int(action["shot_type"])
        if shot_type == 11:
            return action, reasoning
        legal_types = {1, 10} if launch else set(range(2, 10))
        if shot_type not in legal_types or shot_type not in set(available_types):
            return None, reasoning
        return action, reasoning

    def _fallback_action(self, retrieval, launch):
        nearest = self._nearest_case(retrieval["normal_cases"] + retrieval["miss_cases"])
        if nearest is None:
            return {
                "shot_type": 1 if launch else 2,
                "landing_x": 0.0,
                "landing_y": 0.0,
                "moving_x": 0.0,
                "moving_y": 0.0,
            }, "Single-stage fallback used the deterministic legal default.", "fallback_default"

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
            "shot_type": shot_type,
            **coordinates,
        }, "Single-stage fallback copied the nearest legal Twin memory action.", "fallback_nearest_case"

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
        aspects, aspect_source = self._estimate_aspects(
            state_dict,
            state_history,
            profile_instruction,
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

        started_at = time.time()
        retrieval = self.database.retrieve_dual(
            state_dict,
            query_embedding,
            normal_top_k=self.normal_top_k,
            miss_top_k=self.miss_top_k,
            lambda_weight=self.lambda_weight,
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
            state_dict, [], launch=launch, state_history=state_history
        )
        available_types = self.database.available_shot_types(launch=launch)
        user_prompt = self.action_template.render(
            trajectory_text=context["trajectory_text"],
            semantic_query_text=aspects["semantic_query_text"],
            normal_memories=retrieval["normal_cases"],
            miss_memories=retrieval["miss_cases"],
            rally_profile_instruction=profile_instruction,
        )
        started_at = time.time()
        response, raw_llm_output, usage = BaseAgent.get_verified_response_or_none(
            self.llm,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_trials=5,
            validator=lambda candidate: self._parse_action_response(
                candidate, launch, available_types
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
        parsed_action, reasoning = self._parse_action_response(
            response, launch, available_types
        )
        if parsed_action is None:
            parsed_action, reasoning, action_source = self._fallback_action(retrieval, launch)
        else:
            action_source = "llm"

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
        all_case_ids = normal_refs + miss_refs
        self._last_case_ids = all_case_ids
        self.last_decision_trace = {
            "schema_version": 2,
            "architecture": self.retrieval_architecture,
            "pre_stage": {
                "architecture": self.pre_stage_architecture,
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
                "source": aspect_source,
            },
            "retrieval": {
                "normal_case_ids": normal_refs,
                "miss_case_ids": miss_refs,
            },
            "case_ids": all_case_ids,
            "rationale": reasoning,
            "action": {
                "shot_type": shot_type,
                "landing_x": None if shot_type == 11 else landing_x,
                "landing_y": None if shot_type == 11 else landing_y,
                "moving_x": None if shot_type == 11 else moving_x,
                "moving_y": None if shot_type == 11 else moving_y,
            },
            "action_source": f"{aspect_source}+{action_source}",
        }
        return action, reasoning
