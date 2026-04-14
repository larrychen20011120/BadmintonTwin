import numpy as np
import pandas as pd

# phisical
def avg_round(df, match_id, player_name):
    match_df = df[df['match_id'] == match_id]
    return match_df.groupby("rally_id").agg(
        ball_round_max=('ball_round', 'max')
    ).mean().item()

def avg_win_round(df, match_id, player_name):
    match_df = df[df['match_id'] == match_id]
    player_df = match_df[match_df["player"] == player_name]

    win_conditions = ["對手落地判斷失誤", "對手落地致勝", "落點判斷失誤"]
    mask = player_df['lose_reason'].notna() & player_df['lose_reason'].isin(win_conditions)
    win_df = player_df[mask]
    return win_df.groupby("rally_id").agg(
        ball_round_max=('ball_round', 'max')
    ).mean().item()

def avg_lose_round(df, match_id, player_name):
    match_df = df[df['match_id'] == match_id]
    player_df = match_df[match_df["player"] == player_name]

    lose_conditions = ["掛網", "未過網", "犯規", "出界"]
    mask = player_df['lose_reason'].notna() & player_df['lose_reason'].isin(lose_conditions)
    lose_df = player_df[mask]
    return lose_df.groupby("rally_id").agg(
        ball_round_max=('ball_round', 'max')
    ).mean().item()

def avg_moving_distance(df, match_id, player_name):
    match_df = df[df['match_id'] == match_id]
    player_df = match_df[match_df["player"] == player_name]
    return player_df["moving_distance"].mean()
    

def attack_shot_ratio(df, match_id, player_name):
    attack_set = [ '殺球', '切球', '推撲球' ]

    match_df = df[df['match_id'] == match_id]
    player_df = match_df[match_df["player"] == player_name]
    valid_type_df = player_df[~ player_df['type'].str.startswith("接", na=False) ].copy()
    attack_df = valid_type_df[ valid_type_df['type'].isin(attack_set) ].copy()

    return len(attack_df) / len(valid_type_df)

def balance_shot_ratio(df, match_id, player_name):
    balance_set = [ '發短球', '平球', '網前球' ]

    match_df = df[df['match_id'] == match_id]
    player_df = match_df[match_df["player"] == player_name]
    valid_type_df = player_df[~ player_df['type'].str.startswith("接", na=False) ].copy()
    balance_df = valid_type_df[ valid_type_df['type'].isin(balance_set) ].copy()

    return len(balance_df) / len(valid_type_df)

def defence_shot_ratio(df, match_id, player_name):
    defence_set = [ '發長球', '長球', '挑球', '接殺防守' ]

    match_df = df[df['match_id'] == match_id]
    player_df = match_df[match_df["player"] == player_name]
    valid_type_df = player_df[~ player_df['type'].str.startswith("接", na=False) ].copy()
    defence_df = valid_type_df[ valid_type_df['type'].isin(defence_set) ].copy()

    return len(defence_df) / len(valid_type_df)


def avg_opponent_moving_distance(df, match_id, player_name):
    match_df = df[ df['match_id'] == match_id]
    player_df = match_df[match_df["player"] == player_name]
    return player_df["opponent_moving_distance"].mean()


# spatial
def avg_location_x(df, match_id, player_name):
    match_df = df[df['match_id'] == match_id]
    player_df = match_df[match_df["player"] == player_name]

    return player_df["player_location_x"].mean()

def avg_location_y(df, match_id, player_name):
    match_df = df[df['match_id'] == match_id]
    player_df = match_df[match_df["player"] == player_name]

    return player_df["player_location_y"].mean()

def avg_landing_x(df, match_id, player_name):
    match_df = df[df['match_id'] == match_id]
    player_df = match_df[match_df["player"] == player_name]

    return player_df["landing_x"].mean()

def avg_landing_y(df, match_id, player_name):
    match_df = df[df['match_id'] == match_id]
    player_df = match_df[match_df["player"] == player_name]

    return player_df["landing_y"].mean()


"""
extract_match_features: 
  -> extract 12 features for each match_id for a given player_name
"""
def extract_match_features(df, match_ids, player_name):

    player_feature_funcs = [
        avg_round, avg_win_round, avg_lose_round, avg_moving_distance,
        attack_shot_ratio, balance_shot_ratio, defence_shot_ratio, avg_opponent_moving_distance,
        avg_location_x, avg_location_y, avg_landing_x, avg_landing_y
    ]
    
    features_list = []
    
    for mid in match_ids:
        row = {'match_id': mid}
        
        for func in player_feature_funcs:
            try:
                row[func.__name__] = func(df, mid, player_name)
            except Exception as e:
                row[func.__name__] = np.nan
                
        features_list.append(row)
            
    return pd.DataFrame(features_list)

"""
extract_match_features_by_rally: 
  -> For a given match_id and player_name, 
     compute match-level features up to each rally (prefix statistics).
"""
def extract_match_features_by_rally(df, match_id, player_name):

    player_feature_funcs = [
        avg_round, avg_win_round, avg_lose_round, avg_moving_distance,
        attack_shot_ratio, balance_shot_ratio, defence_shot_ratio,
        avg_opponent_moving_distance,
        avg_location_x, avg_location_y, avg_landing_x, avg_landing_y
    ]

    match_df = df[df['match_id'] == match_id]
    rally_ids = sorted(match_df['rally_id'].unique())

    features_list = []

    # iterate through all rally ids
    for rid in rally_ids:
        
        prefix_df = match_df[match_df['rally_id'] < rid]

        row = {
            'match_id': match_id,
            'rally_id': rid
        }

        for func in player_feature_funcs:
            try:
                row[func.__name__] = func(prefix_df, match_id, player_name)
            except Exception:
                row[func.__name__] = np.nan

        features_list.append(row)

    return pd.DataFrame(features_list)