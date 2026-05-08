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
                 teacher_forcing=False, save_trajectories=False):
        """
        Args:
            player_name: 選手完整名稱
            experiment: 實驗類型 ('norm' 或 'shifted')
            device: 運算裝置
            skip_first_n: 跳過前n個shots (只在存檔時記錄，不影響模擬)
            teacher_forcing: 是否使用 teacher forcing（餵 GT state）模式
            save_trajectories: 是否儲存原始軌跡（雙方選手、完整 GT）
        """
        self.player_name = player_name
        self.experiment = experiment
        self.device = device
        self.skip_first_n = skip_first_n
        self.teacher_forcing = teacher_forcing
        self.save_trajectories = save_trajectories
        
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
        output_path = os.path.join(
            'Experiments', algo, 'trajectories',
            f"{safe_name}_{self.experiment}.pkl"
        )
        save_trajectories(rally_trajectories, output_path)
