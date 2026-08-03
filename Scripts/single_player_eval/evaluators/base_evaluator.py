"""
Base Evaluator Class for Single Player Evaluation
"""

from abc import ABC, abstractmethod
import json
import os

from Utils.llm import LLM
from Utils.runtime_report import algorithm_runtime_rows, llm_rows, print_runtime_box


def resolve_prompt_variant(experiment, style_shift_instruction_mode):
    """Select the shifted prompt only when style-shift instructions are enabled."""
    if style_shift_instruction_mode not in ('auto', 'off'):
        raise ValueError("style_shift_instruction_mode must be 'auto' or 'off'")
    if experiment == 'shifted' and style_shift_instruction_mode == 'auto':
        return 'shifted'
    return 'norm'


class BaseEvaluator(ABC):
    """
    單一選手評估的基礎類別
    
    所有演算法的evaluator都應該繼承這個類別
    """
    
    def __init__(self, player_name, experiment='norm', device='cuda', skip_first_n=0,
                 save_trajectories=False,
                 style_shift_instruction_mode='auto', verbose_style_shift=False,
                 output_experiment=None):
        """
        Args:
            player_name: 選手完整名稱
            experiment: 實驗類型 ('norm' 或 'shifted')
            device: 運算裝置
            skip_first_n: 跳過前n個shots (只在存檔時記錄，不影響模擬)
            save_trajectories: 是否儲存原始軌跡（雙方選手、完整 GT）
            style_shift_instruction_mode: 'auto' 會使用 shifted prompt 並注入可用
                instruction；'off' 會使用一般 prompt 並關閉注入
            verbose_style_shift: 是否印出 style-shift instruction 套用狀態
            output_experiment: 存檔用的 experiment label；不影響 split/data path
        """
        self.player_name = player_name
        self.experiment = experiment
        self.output_experiment = output_experiment or experiment
        self.device = device
        self.skip_first_n = skip_first_n
        self.save_trajectories = save_trajectories
        self.trajectory_output_path = None
        self.trajectory_run_metadata = {}
        self.style_shift_instruction_mode = style_shift_instruction_mode
        self.prompt_variant = resolve_prompt_variant(
            experiment,
            style_shift_instruction_mode,
        )
        self.verbose_style_shift = verbose_style_shift
        self._style_shift_instruction_map = None
        self._style_shift_debug_seen = set()
        
        # 載入config
        self.config = self._load_config()
        
        # 設定路徑
        self.test_split_path = (
            self.config['paths']['rally_split']
            if experiment == 'rally'
            else self.config['paths'][f'{experiment}_split']
        )
        self.merged_csv_path = self.config['paths']['merged_csv']
        self.player_map_path = self.config['paths']['player_map']
        self.rally_profiles_path = self.config['paths'].get('rally_profiles')

    def print_runtime_parameters(self, algorithm, extra_rows=None, include_llm=False, **effective):
        rows = algorithm_runtime_rows(
            self.config,
            algorithm,
            experiment=self.experiment,
            player=self.player_name,
            device=self.device,
            skip_first_n=self.skip_first_n,
            **effective,
        )

        if extra_rows:
            rows.extend(extra_rows)

        if include_llm:
            algo_config = self.config.get(algorithm, {})
            settings = LLM.resolve_effective_settings(
                model_name=algo_config.get('model_name'),
                api_key=algo_config.get('api_key'),
                host=algo_config.get('host'),
                port=algo_config.get('port'),
                base_url=algo_config.get('base_url'),
                host_url=algo_config.get('host_url'),
            )
            rows.extend(llm_rows(settings))

        print_runtime_box(f"Runtime Parameters: {algorithm}", rows)
        
    def _load_config(self):
        """載入設定檔"""
        import yaml
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            'config.yaml'
        )
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_player_test_matches(self):
        """
        取得指定選手的測試比賽IDs
        
        Returns:
            list: match_id列表
        """
        with open(self.test_split_path, 'r') as f:
            test_split = json.load(f)
        
        test_data = test_split.get('test', {})
        
        if self.player_name not in test_data:
            available_players = list(test_data.keys())
            raise ValueError(
                f"選手 '{self.player_name}' 不在test split中。\n"
                f"可用的選手: {available_players}"
            )
        
        return test_data[self.player_name]

    def build_rollout_info(self, match_id, rally_num, current_state, action=None):
        """Build agent info from rollout state instead of ground-truth rows."""
        return {
            'match_id': match_id,
            'rally': rally_num,
            'round': [int(current_state[1].item())],
            'score': [int(current_state[2].item()), int(current_state[3].item())],
            'action': [action]
        }

    def _project_path(self):
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    def _load_style_shift_instruction_map(self):
        """Lazy-load (player_name, match_id) -> style-shift instruction."""
        if self._style_shift_instruction_map is not None:
            return self._style_shift_instruction_map

        style_shift_path = self.config['paths'].get('style_shift_json')
        if style_shift_path and not os.path.isabs(style_shift_path):
            style_shift_path = os.path.join(self._project_path(), style_shift_path)

        instruction_map = {}
        if style_shift_path and os.path.exists(style_shift_path):
            with open(style_shift_path, 'r', encoding='utf-8') as f:
                style_data = json.load(f)

            for player_data in style_data.get('shifted', []):
                player_name = player_data.get('name')
                match_ids = player_data.get('match_ids', [])
                instructions = player_data.get('instructions', [])
                if not player_name:
                    continue

                for match_id, instruction in zip(match_ids, instructions):
                    if instruction:
                        instruction_map[(player_name, int(match_id))] = instruction

        self._style_shift_instruction_map = instruction_map
        return self._style_shift_instruction_map

    def get_available_style_shift_instruction(self, player_name, match_id):
        """Return the instruction in style_shift.json, ignoring auto/off mode."""
        if player_name is None or match_id is None:
            return ''

        try:
            match_id = int(match_id)
        except (TypeError, ValueError):
            return ''

        return self._load_style_shift_instruction_map().get((player_name, match_id), '')

    def get_style_shift_instruction(self, player_name, match_id):
        """Return the instruction only when mode allows injection."""
        if self.style_shift_instruction_mode == 'off':
            return ''
        return self.get_available_style_shift_instruction(player_name, match_id)

    def attach_style_shift_instruction(self, info, player_name, row):
        """Attach the current player's style-shift instruction to info, if enabled."""
        match_id = row.get('match_id') if hasattr(row, 'get') else None
        available_instruction = self.get_available_style_shift_instruction(player_name, match_id)
        instruction = self.get_style_shift_instruction(player_name, match_id)

        info['match_id'] = match_id
        info['style_shift_instruction'] = instruction
        info['style_shift_instruction_player'] = player_name if instruction else ''
        info['style_shift_instruction_mode'] = self.style_shift_instruction_mode

        self._log_style_shift_instruction(player_name, row, instruction, available_instruction)
        return instruction

    def attach_style_shift_instruction_for_rollout(self, info, player_name, match_id, rally_num=None):
        """Attach style-shift instruction when rollout has no current GT row."""
        available_instruction = self.get_available_style_shift_instruction(player_name, match_id)
        instruction = self.get_style_shift_instruction(player_name, match_id)

        info['match_id'] = match_id
        info['style_shift_instruction'] = instruction
        info['style_shift_instruction_player'] = player_name if instruction else ''
        info['style_shift_instruction_mode'] = self.style_shift_instruction_mode

        row = {
            'match_id': match_id,
            'rally': rally_num if rally_num is not None else info.get('rally', ''),
            'ball_round': info.get('round', [''])[-1] if info.get('round') else '',
        }
        self._log_style_shift_instruction(player_name, row, instruction, available_instruction)
        return instruction

    def _log_style_shift_instruction(self, player_name, row, instruction, available_instruction):
        if not self.verbose_style_shift or self.experiment != 'shifted':
            return

        match_id = row.get('match_id') if hasattr(row, 'get') else None
        try:
            match_id_for_key = int(match_id)
        except (TypeError, ValueError):
            match_id_for_key = match_id

        if instruction:
            status = 'APPLY'
        elif self.style_shift_instruction_mode == 'off' and available_instruction:
            status = 'SKIP'
        else:
            status = 'NONE'

        key = (status, player_name, match_id_for_key, self.style_shift_instruction_mode)
        if key in self._style_shift_debug_seen:
            return
        self._style_shift_debug_seen.add(key)

        rally = row.get('rally', '') if hasattr(row, 'get') else ''
        ball_round = row.get('ball_round', '') if hasattr(row, 'get') else ''
        prefix = (
            f"[StyleShift] {status} algorithm={self.get_algorithm_name()} "
            f"mode={self.style_shift_instruction_mode} match_id={match_id} "
            f"rally={rally} round={ball_round} player=\"{player_name}\""
        )

        if status == 'APPLY':
            print(prefix)
            print(f"[StyleShift] instruction=\"{instruction}\"")
        elif status == 'SKIP':
            print(f"{prefix} reason=disabled")
            print(f"[StyleShift] available_instruction=\"{available_instruction}\"")
        else:
            print(prefix)
    
    @abstractmethod
    def simulate(self):
        """
        執行模擬並收集trajectory
        
        Returns:
            list: trajectory list，每個element是一個rally dict
        """
        pass
    
    @abstractmethod
    def get_algorithm_name(self):
        """取得演算法名稱"""
        pass
    
    def _save_trajectories_if_needed(self, rally_trajectories):
        """Save one JSON object per rally if --save-trajectories is set."""
        if not self.save_trajectories:
            return
        from Scripts.single_player_eval.trajectory_store import (
            create_run_directory,
            save_manifest,
            save_trajectories,
        )

        algo = self.get_algorithm_name()
        metadata = {
            'algorithm': algo,
            'experiment': self.experiment,
            'output_experiment': self.output_experiment,
            **self.trajectory_run_metadata,
        }
        output_dir = self.trajectory_output_path
        if output_dir is None:
            output_root = os.path.join(
                'Experiments', algo, 'trajectories', self.output_experiment
            )
            run_dir, generation = create_run_directory(output_root)
            output_dir = os.path.join(run_dir, 'rallies')
            metadata['generation'] = generation
            save_manifest(run_dir, metadata)

        save_trajectories(rally_trajectories, output_dir, metadata)
