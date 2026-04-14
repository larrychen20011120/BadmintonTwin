import os
import sys
import numpy as np

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Agent.base_agent import BaseAgent

class RandomAgent(BaseAgent):

    def __init__(self, player_name):
        super().__init__(player_name)

    def predict(self, state, info, launch):
        
        hit = state[-1]
        output = np.random.rand(4,) * 2 - 1
        output_shot = np.random.randint(11, size=1)[0] + 1

        uniform_dist = [0.1] * 10
        action = (output_shot, hit, tuple(output[:2]), tuple(output[2:]), uniform_dist)

        return action, ""

