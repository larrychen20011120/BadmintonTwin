"""
Pure LLM Evaluator for Single Player.
"""

import json
import pandas as pd

from .base_evaluator import BaseEvaluator
from .naive_rag_evaluator import NaiveRAGEvaluator
from Agent.pure_llm import PureLLMAgent
from Utils.helper import state_col, action_col


class PureLLMEvaluator(NaiveRAGEvaluator):
    """Single-player evaluator for the retrieval-free LLM baseline."""

    def __init__(self, player_name, experiment='norm', device='cuda', skip_first_n=0,
                 enable_tracing=True, save_trajectories=False,
                 style_shift_instruction_mode='auto', verbose_style_shift=False,
                 output_experiment=None):
        BaseEvaluator.__init__(
            self,
            player_name,
            experiment,
            device,
            skip_first_n,
            save_trajectories=save_trajectories,
            style_shift_instruction_mode=style_shift_instruction_mode,
            verbose_style_shift=verbose_style_shift,
            output_experiment=output_experiment,
        )

        self.enable_tracing = enable_tracing

        with open(self.player_map_path, 'r') as f:
            self.player2index = json.load(f)

        if self.player_name not in self.player2index:
            raise ValueError(f"Player '{self.player_name}' not found in player map")

        self.player_id = self.player2index[self.player_name]

        self.df = pd.read_csv(self.merged_csv_path)
        target_cols = state_col + action_col
        self.df = self.df.dropna(subset=target_cols).reset_index(drop=True)

        self.agents = {}
        self.agent = self._get_agent(self.player_name)

        self.print_runtime_parameters(
            "pure_llm",
            include_llm=True,
            extra_rows=[
                ("prompt_variant", self.prompt_variant),
                ("prompt_mode", "zero_shot"),
                ("prompt_dir", "Prompts/New/pure_llm"),
            ],
        )

    def _create_agent(self, player_name, player_id):
        llm_config = self.config.get('pure_llm', {})
        agent = PureLLMAgent(
            player_name=player_name,
            model_name=llm_config.get('model_name'),
            api_key=llm_config.get('api_key'),
            host=llm_config.get('host'),
            port=llm_config.get('port'),
            base_url=llm_config.get('base_url'),
            host_url=llm_config.get('host_url'),
            enable_tracing=self.enable_tracing,
            prompt_variant=self.prompt_variant,
        )
        print(f"[PureLLM] Agent created for {player_name}")
        return agent

    def get_algorithm_name(self):
        return "pure_llm"
