import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def merge_shuttleset(shuttleset_path, shuttleset22_path, save_path):

    shuttleset_df = pd.read_csv(shuttleset_path)
    shuttleset22_df = pd.read_csv(shuttleset22_path)

    # map correct type to constraint definition
    t2c = {1: 1, 2: 10, 3: 2, 4: 4, 5: 9, 6: 8, 7: 6, 8: 7, 9: 3, 10: 5, 11: 11, 0: 0}
    tw2index= {
        "發短球": 1,
        "發長球": 10,
        "長球": 2,
        "推撲球": 3,
        "殺球": 4,
        "接殺防守": 5,
        "平球": 6,
        "接不到": 11,
        "網前球": 7,
        "挑球": 8,
        "切球": 9,
    }

    # Adjust the order of the dataframes
    shuttleset_df = shuttleset_df.sort_values(by=['match_id', 'set', 'rally', 'ball_round'])
    shuttleset_df = shuttleset_df.reset_index(drop=True)

    shuttleset22_df = shuttleset22_df.sort_values(by=['match_id', 'set', 'rally', 'ball_round'])
    shuttleset22_df = shuttleset22_df.reset_index(drop=True)

    shuttleset22_df["match_id"] += 44
    merged_df = pd.concat(
        [shuttleset_df, shuttleset22_df], axis=0, ignore_index=True
    )

    merged_df["match_id"] = merged_df["match_id"].astype(int)
    merged_df["player_type"] = merged_df["player_type"].map(t2c)
    merged_df["opponent_type"] = merged_df["opponent_type"].map(t2c)

    for i in tqdm(range(1, len(merged_df))):
        prev = merged_df.loc[i-1]

        win_conditions = ["對手落地判斷失誤", "對手落地致勝", "落點判斷失誤"]
        if pd.notna(prev['lose_reason']) and prev['lose_reason'] in win_conditions:
            merged_df.loc[i, "hit_x"] = prev["landing_x"]
            merged_df.loc[i, "hit_y"] = prev["landing_y"]
            merged_df.loc[i, "hit_area"] = prev["landing_area"]
            merged_df.loc[i, "score_status"] = prev["score_status"]
            merged_df.loc[i, "player_score"] = prev["player_score"]
            merged_df.loc[i, "opponent_score"] = prev["opponent_score"]
            merged_df.loc[i, "player_location_area"] = prev["opponent_location_area"]
            merged_df.loc[i, "player_location_x"] = prev["opponent_location_x"]
            merged_df.loc[i, "player_location_y"] = prev["opponent_location_y"]
            merged_df.loc[i, "opponent_location_area"] = prev["player_move_area"]
            merged_df.loc[i, "opponent_location_x"] = prev["moving_x"]
            merged_df.loc[i, "opponent_location_y"] = prev["moving_y"]
            merged_df.loc[i, "ball_distance_x"] = merged_df.loc[i, "hit_x"] - merged_df.loc[i, "player_location_x"]
            merged_df.loc[i, "ball_distance_y"] = merged_df.loc[i, "hit_y"] - merged_df.loc[i, "player_location_y"]
            merged_df.loc[i, "ball_distance"] = np.sqrt(merged_df.loc[i, "ball_distance_x"]**2 + merged_df.loc[i, "ball_distance_y"]**2)
            merged_df.loc[i, "opponent_type"] = prev["player_type"]
            merged_df.loc[i, "player_type"] = 11
            merged_df.loc[i, "type"] = "接不到"
        
        if pd.notna(merged_df.loc[i, 'server']) and merged_df.loc[i, 'server'] == 3:
            merged_df.loc[i, "player_type"] = tw2index[merged_df.loc[i,"type"]]
            merged_df.loc[i, "opponent_type"] = prev["player_type"]

    lose_conditions = ["掛網", "未過網", "犯規"]
    mask = merged_df['lose_reason'].notna() & merged_df['lose_reason'].isin(lose_conditions)
    merged_df.loc[mask, "player_type"] = 11
    merged_df.loc[mask, "server"] = 3
    drop_mask = mask.shift(1, fill_value=False)
    merged_df = merged_df[~drop_mask].reset_index(drop=True)

    merged_df["opponent_moving_distance_x"] = merged_df["landing_x"] - merged_df["opponent_location_x"]
    merged_df["opponent_moving_distance_y"] = merged_df["landing_y"] - merged_df["opponent_location_y"]
    merged_df["moving_distance"] = np.sqrt( merged_df["player_move_x"]**2 + merged_df["player_move_y"] **2 )
    merged_df["opponent_moving_distance"] = np.sqrt( merged_df["opponent_moving_distance_x"]**2 + merged_df["opponent_moving_distance_y"] **2 )

    merged_df.loc[merged_df[merged_df["type"] == "接不到"].index, "opponent_moving_distance"] = 0 

    merged_df.to_csv(save_path)
    print("Total size: ", len(merged_df))
    print(f"Save the merged file to {save_path}!")

    return merged_df




if __name__ == "__main__":

    shuttleset_path = os.path.join("Data", "input_data", "shuttleset.csv")
    shuttleset22_path = os.path.join("Data", "input_data", "shuttleset22.csv")

    merged_target_path = os.path.join("Data", "input_data", "merged.csv")
    match_target_path = os.path.join("Data", "input_data", "match_player.csv")

    shuttlesets_df = merge_shuttleset(shuttleset_path, shuttleset22_path, merged_target_path)