"""
RallyNet Evaluator for Single Player
"""

import os
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_path)

from .base_evaluator import BaseEvaluator


class RallyNetEvaluator(BaseEvaluator):
    """RallyNet單一選手評估器"""
    
    def __init__(self, player_name, experiment='norm', device='cuda', skip_first_n=0,
                 model_path=None, hyperparameters_path=None):
        """
        Args:
            player_name: 選手名稱
            experiment: 實驗類型
            device: 運算裝置
            skip_first_n: 跳過前n個shots
            model_path: 模型路徑（可選）
            hyperparameters_path: hyperparameters路徑（可選）
        """
        super().__init__(player_name, experiment, device, skip_first_n)
        
        # RallyNet目前沒有在config中定義路徑，使用預設值
        if model_path is None:
            self.model_path = f'Experiments/rallynet/weight/gen_e_30000.trc'
        else:
            self.model_path = model_path
        
        if hyperparameters_path is None:
            self.hyperparameters_path = 'Experiments/rallynet/hyperparameters.pkl'
        else:
            self.hyperparameters_path = hyperparameters_path
    
    def get_algorithm_name(self):
        return "rallynet"
    
    def evaluate(self):
        """
        執行RallyNet評估
        
        這裡調用原有的eval_rallynet.py中的評估函數，
        但只針對單一選手
        """
        # TODO: 實現RallyNet評估邏輯
        # 目前先回傳placeholder，實際實現時需要參考Agent/eval_rallynet.py
        
        raise NotImplementedError(
            "RallyNet evaluator not yet implemented. "
            "Please use Agent/eval_rallynet.py directly or implement this method."
        )
