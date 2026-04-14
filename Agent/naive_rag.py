import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from jinja2 import Template

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Agent.base_agent import BaseAgent
from Agent.llm import LLM


def vectorize(traj):
    # traj: [shot_type, player_loc, opponent_loc, ball_loc]
    type_count = 11
    vector = [ 0 for _ in range(type_count) ]
    vector[traj[0]-1] = 1
    vector.extend( [ coord for point in traj[1:] for coord in point ] )
    return vector



class DataBase:
    def __init__(self, df, match_ids, player_name):
        # select the df whose match_id in match_ids
        self.df = df[df['match_id'].isin(match_ids)]
        self.player_name = player_name
        
    def retrieve(self, rally_trajectory, k=10):
        pass

    def distance(self, traj1, traj2):
        

    
class NaiveRAGAgent(BaseAgent):
    
    def __init__(self, player_name, df, model_name, api_key="EMPTY", style_shift_inst=""):

        super().__init__(player_name)

        self.llm = LLM(model_name=model_name, api_key=api_key)
        self.style_shift_inst = style_shift_inst

        self.round_digits = 4
        self.t = 0
        self.k = 12

        self.db = DataBase(df, df['match_id'].unique().tolist(), player_name)
        
    def action(self, state, info, launch):


        return (output_shot, hit, tuple(output_land), tuple(output_move), normal_dist), reason



class TimeRAGAgent(BaseAgent):
    
    def __init__(self, player_name, df, model_name, api_key="EMPTY", style_shift_inst=""):

        super().__init__(player_name)

        self.llm = LLM(model_name=model_name, api_key=api_key)
        self.style_shift_inst = style_shift_inst

        self.round_digits = 4
        self.t = 0
        self.k = 12


        self.col_names = [
            "player_location_x", "player_location_y", 
            "hit_x", "hit_y", 
            "opponent_location_x", "opponent_location_y", 
            "pre_type"
        ]
        self.discrete_dict = {
            "pre_type": 12,
            "type": 12
        }

        self.type2index = {
            '發短球': 1, '發長球': 2, 
            '推撲球': 3, '殺球': 4, '接殺防守': 5, '平球': 6, 
            '網前球': 7, '挑球': 8, '切球': 9, '長球': 10,
            "11": 11, "0": 0
        }
        self.index2type = {
            v: k
            for k, v in self.type2index.items()
        }
        # Read the temlplate
        ### - system prompt 
        ### - two stage user prompts 
        ### - reason column construction prompt
        with open("BadmintonGPT/Prompts/system.txt", "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
        with open("BadmintonGPT/Prompts/situation_history.template", "r", encoding="utf-8") as f:
            self.situation_template = Template(f.read())
        with open("BadmintonGPT/Prompts/stage1.template", "r", encoding="utf-8") as f:
            self.stage1_template = Template(f.read())
        with open("BadmintonGPT/Prompts/stage2.template", "r", encoding="utf-8") as f:
            self.stage2_template = Template(f.read())
        with open("BadmintonGPT/Prompts/stage2_situation.template", "r", encoding="utf-8") as f:
            self.stage2_situation_template = Template(f.read())
        with open("BadmintonGPT/Prompts/reason.template", "r", encoding="utf-8") as f:
            self.reason_template = Template(f.read())

        # Read the schema verifier
        with open("BadmintonGPT/Schemas/stage1.json", "r", encoding="utf-8") as f:
            self.stage1_schema = json.load(f)
        with open("BadmintonGPT/Schemas/stage2.json", "r", encoding="utf-8") as f:
            self.stage2_schema = json.load(f)
        with open("BadmintonGPT/Schemas/situation.json", "r", encoding="utf-8") as f:
            self.situation_schema = json.load(f)

        self.__create_db(player, df)

        
    def action(self, state, info, launch):


        return (output_shot, hit, tuple(output_land), tuple(output_move), normal_dist), reason


    def __retrieve(self, query_vec, k=10, selected_indices=None):

        pass
    

    def __vectorize_row(self, row):
        pass
    
    
    def __create_db(self, player, df):

        self.player_df = df.groupby('player').get_group(player)
        self.player_df = self.player_df.reset_index(drop=True)

        # target macro following    
        ### historical statistics
        self.shot_type_statistics = {
            self.index2type[k]: v
            for k, v in self.player_df["type"].value_counts(normalize=True).items()
        }
        self.target_error_rate = self.shot_type_statistics.get("11", 0.0)
        self.shot_type_statistics.pop("11", None)
        ### Recalculate to make sum to 1.0
        total = sum(self.shot_type_statistics.values())
        self.shot_type_statistics = {
            k: round(v / total, self.round_digits)
            for k, v in self.shot_type_statistics.items()
        }
        

        if len(self.situation) != 0:

            # render prompt
            user_prompt = self.situation_template.render(
                shot_dist=self.shot_type_statistics,
                situation=self.situation
            )

            data = self.llm.get_verified_response(
                schema=self.situation_schema,
                user_prompt=user_prompt
            )

            if data is not None:
                self.shot_type_statistics = {
                    k: json.loads(v) if isinstance(v, str) else v
                    for k, v in data["新球種分布"].items()
                }

        self.curr_state["shot_dist"] = {
            shot_type: 0.0
            for shot_type in self.shot_type_statistics.keys()
        }
        self.shot_type_indices = {
            shot_type: list(self.player_df[self.player_df["type"] == self.type2index[shot_type]].index)
            for shot_type in self.shot_type_statistics.keys()
        }
        
        matrix = []
        for _, row in self.player_df.iterrows():
            matrix.append(self.__vectorize_row(row))

        self.db = np.array(matrix)

