import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from jinja2 import Template

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Preprocessing.match_feature import extract_match_features
from Utils.llm import LLM


def construct_user_prompt(user_prompt_template, normal_profile, abnormal_profile):

    name_map = {
        'avg_round': "average number of shots per rally",
        'avg_win_round': "average number of shots in rallies won by the player",
        'avg_lose_round': "average number of shots in rallies lost by the player", 
        'avg_moving_distance': "average distance traveled by the player per shot",
        'attack_shot_ratio': "the ratio of attacking shots in the match", 
        'balance_shot_ratio': "the ratio of balance shots in the match", 
        'defence_shot_ratio': "the ratio of defensive shots in the match", 
        'avg_opponent_moving_distance': "average distance traveled by the opponent per shot",
        'avg_location_x': "average player position on the court (X-axis)",
        'avg_location_y': "average player position on the court (Y-axis)",
        'avg_landing_x': "average landing position of the shuttlecock on the opponent court (X-axis)", 
        'avg_landing_y': "average landing position of the shuttlecock on the opponent court (Y-axis)", 
    }
    data = {
        "normal_profile":       { name_map[k]: v for k, v in normal_profile.items() },
        "abnormal_profile":     { name_map[k]: v for k, v in abnormal_profile.items() }
    }
    user_prompt = user_prompt_template.render(
		normal_profile=data["normal_profile"],
		abnormal_profile=data["abnormal_profile"]
	)
    return user_prompt
	
def get_difference(llm, schema, system_prompt="", user_prompt=""):
	response = llm.get_verified_response(schema, system_prompt, user_prompt)
	print(response)
	return response["overall_style_shift_summary"]


def get_iqr_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

# def get_std_outliers(series, threshold=3):
#     mean = series.mean()
#     std = series.std()
    
#     lower_bound = mean - threshold * std
#     upper_bound = mean + threshold * std
    
#     return (series < lower_bound) | (series > upper_bound)

# def get_mad_outliers(series, threshold=3):
#     median = series.median()
#     mad = (series - median).abs().median()
#     modified_z_score = 0.6745 * (series - median) / mad
    
#     return modified_z_score.abs() > threshold


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--enough_thres", type=int, default=7, help="Over this threshold, we need to calculate normal set or anomaly set.")
    parser.add_argument("--outlier_thres", type=int, default=2, help="Number of features are outliers!")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--base_url", type=str, default=None)

    args = parser.parse_args()

    merged_path = os.path.join("Data", "input_data", "merged.csv")
    match_path = os.path.join("Data", "input_data", "match_player.csv")
    user_template_path = os.path.join("Prompts", "Preprocessing", "style_shift.jinja")
    shcema_path = os.path.join("Schemas", "Preprocessing", "style_shift_schema.json")
    

    merged_df = pd.read_csv(merged_path)
    match_df = pd.read_csv(match_path)

    with open(user_template_path, "r", encoding="utf-8") as f:
        user_prompt_template = Template(f.read())
    with open(shcema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    llm = LLM(model_name=args.model_name, api_key=args.api_key, base_url=args.base_url)

    player_counts = match_df['player'].value_counts()
    opponent_counts = match_df['opponent'].value_counts()
    total_counts = player_counts.add(opponent_counts, fill_value=0).astype(int)

    enough_players = total_counts[total_counts >= args.enough_thres].index
    not_enough_players = total_counts[total_counts < args.enough_thres].index

    data = {
        "normal": [],
        "shifted": []
    }
    # each data item is { 'name', 'match_ids' }

    for player_name in not_enough_players:

        match_ids = match_df[ (match_df['player'] == player_name) | (match_df['opponent'] == player_name) ]["match_id"]
        feature_df = extract_match_features(merged_df, match_ids, player_name)

        features_cols = feature_df.columns.difference(['match_id'])
        match_profile = dict(feature_df[features_cols].mean())
        match_profile = {
            k: v.item()
            for k, v in match_profile.items()
        }

        normal_item = {
            "name": player_name,
            "match_ids": match_ids.tolist(),
            "match_profile": match_profile
        }
        data["normal"].append(normal_item)


    for player_name in enough_players:

        match_ids = match_df[ (match_df['player'] == player_name) | (match_df['opponent'] == player_name) ]["match_id"]
        feature_df = extract_match_features(merged_df, match_ids, player_name)

        # get the feature names (exclude 'match_id')
        features_cols = feature_df.columns.difference(['match_id'])

        is_outlier_iqr = feature_df[features_cols].apply(get_iqr_outliers)
        # is_outlier_iqr = feature_df[features_cols].apply(get_mad_outliers)
        feature_df['outlier_count'] = is_outlier_iqr.sum(axis=1)

        features_cols = feature_df.columns.difference(['match_id', 'outlier_count'])

        normal_df = feature_df[feature_df['outlier_count'] < args.outlier_thres]
        abnormal_df = feature_df[feature_df['outlier_count'] >= args.outlier_thres]

        normal_profile = dict(normal_df[features_cols].mean())
        normal_profile = {
            k: v.item()
            for k, v in normal_profile.items()
        }

        normal_item = {
            "name": player_name,
            "match_ids": normal_df["match_id"].tolist(),
            "match_profile": normal_profile
        }
        data["normal"].append(normal_item)

        match_profiles = []
        ids = []
        instructions = []

        for id in abnormal_df["match_id"]:
            abnormal_profile = dict(abnormal_df[abnormal_df["match_id"] == id][features_cols])

            need_append = True

            for k, v in abnormal_profile.items():
                if np.isnan(v).item():
                    need_append = False
                    break        

            abnormal_profile = {
                k: v.item()
                for k, v in abnormal_profile.items()
            }

            if need_append:
                user_prompt = construct_user_prompt(user_prompt_template, normal_profile, abnormal_profile)
                instruction = get_difference(llm, schema, user_prompt=user_prompt)

                match_profiles.append(abnormal_profile)
                ids.append(id)
                instructions.append(instruction)

        abnormal_item = {
            "name": player_name,
            "match_ids": ids,
            "match_profiles": match_profiles,
            "instructions": instructions
        }

        data["shifted"].append(abnormal_item)
        
    # save the normal and style_shifted data
    save_path = os.path.join("Data", "intermediate_data", "style_shift.json")
    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("JSON File Dumped!!")
