"""
Base Evaluator Class for Single Player Evaluation
"""

from abc import ABC, abstractmethod
import json
import os


class BaseEvaluator(ABC):
    """
    單一選手評估的基礎類別
    
    所有演算法的evaluator都應該繼承這個類別
    """
    
    def __init__(self, player_name, experiment='norm', device='cuda', skip_first_n=0,
                 teacher_forcing=False, save_trajectories=False,
                 style_shift_instruction_mode='auto', verbose_style_shift=False,
                 output_experiment=None):
        """
        Args:
            player_name: 選手完整名稱
            experiment: 實驗類型 ('norm' 或 'shifted')
            device: 運算裝置
            skip_first_n: 跳過前n個shots (只在存檔時記錄，不影響模擬)
            teacher_forcing: 是否使用 teacher forcing（餵 GT state）模式
            save_trajectories: 是否儲存原始軌跡（雙方選手、完整 GT）
            style_shift_instruction_mode: 'auto' 會注入可用 instruction；'off' 會關閉注入
            verbose_style_shift: 是否印出 style-shift instruction 套用狀態
            output_experiment: 存檔用的 experiment label；不影響 split/data path
        """
        if style_shift_instruction_mode not in ('auto', 'off'):
            raise ValueError("style_shift_instruction_mode must be 'auto' or 'off'")

        self.player_name = player_name
        self.experiment = experiment
        self.output_experiment = output_experiment or experiment
        self.device = device
        self.skip_first_n = skip_first_n
        self.teacher_forcing = teacher_forcing
        self.save_trajectories = save_trajectories
        self.trajectory_output_path = None
        self.style_shift_instruction_mode = style_shift_instruction_mode
        self.verbose_style_shift = verbose_style_shift
        self._style_shift_instruction_map = None
        self._style_shift_debug_seen = set()
        
        # 載入config
        self.config = self._load_config()
        
        # 設定路徑
        self.test_split_path = self.config['paths'][f'{experiment}_split']
        self.merged_csv_path = self.config['paths']['merged_csv']
        self.player_map_path = self.config['paths']['player_map']
        
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
        """Save rally trajectory dicts to pickle if --save-trajectories is set."""
        if not self.save_trajectories:
            return
        from Scripts.single_player_eval.trajectory_store import save_trajectories

        safe_name = self.player_name.replace(' ', '_')
        algo = self.get_algorithm_name()
        output_path = self.trajectory_output_path or os.path.join(
            'Experiments', algo, 'trajectories',
            f"{safe_name}_{self.output_experiment}.pkl"
        )
        save_trajectories(rally_trajectories, output_path)
