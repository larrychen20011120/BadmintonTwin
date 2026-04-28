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
    
    def __init__(self, player_name, experiment='norm', device='cuda', skip_first_n=0):
        """
        Args:
            player_name: 選手完整名稱
            experiment: 實驗類型 ('norm' 或 'shifted')
            device: 運算裝置
            skip_first_n: 跳過前n個shots
        """
        self.player_name = player_name
        self.experiment = experiment
        self.device = device
        self.skip_first_n = skip_first_n
        
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
            os.path.dirname(os.path.dirname(__file__)), 
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
    def evaluate(self):
        """
        執行評估
        
        Returns:
            dict: 評估結果，格式與現有eval結果相同
            {
                'player_name': {
                    'land_dtw': float,
                    'move_dtw': float, 
                    'shot_ctc': float,
                    'edit_dist': float,
                    'num_rallies': int,
                    'num_shots': int
                }
            }
        """
        pass
    
    @abstractmethod
    def get_algorithm_name(self):
        """取得演算法名稱"""
        pass
    
    def save_results(self, results, output_dir=None):
        """
        儲存評估結果
        
        Args:
            results: 評估結果字典
            output_dir: 輸出目錄（可選）
        
        Returns:
            str: 儲存的檔案路徑
        """
        if output_dir is None:
            # 預設路徑: Experiments/{algorithm}/results/single_player/
            algorithm = self.get_algorithm_name()
            output_dir = os.path.join(
                'Experiments', algorithm, 'results', 'single_player'
            )
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 檔名格式: {player_name}_{experiment}_eval.json
        safe_player_name = self.player_name.replace(' ', '_')
        filename = f"{safe_player_name}_{self.experiment}_eval.json"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return output_path
    
    def print_results(self, results):
        """
        印出評估結果
        
        Args:
            results: 評估結果字典
        """
        import numpy as np
        
        print("\n" + "="*80)
        print(f"{self.get_algorithm_name().upper()} Single Player Evaluation Results")
        print("="*80)
        print(f"Player: {self.player_name}")
        print(f"Experiment: {self.experiment}")
        print(f"Skip First N: {self.skip_first_n}")
        print("-"*80)
        
        player_data = results.get(self.player_name, {})
        
        if player_data.get('num_rallies', 0) > 0:
            print(f"Land DTW:    {player_data['land_dtw']:.4f}")
            print(f"Move DTW:    {player_data['move_dtw']:.4f}")
            print(f"Shot CTC:    {player_data['shot_ctc']:.4f}")
            print(f"Edit Dist:   {player_data['edit_dist']:.4f}")
            print(f"Rallies:     {player_data['num_rallies']}")
            print(f"Shots:       {player_data['num_shots']}")
        else:
            print("No rallies evaluated.")
        
        print("="*80)
