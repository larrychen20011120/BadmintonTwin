import pandas as pd
import random

if __name__ == "__main__":
    # Instantiate and train the model

    # Train the model using data from a CSV file
    # model.train_model('/home/adsl-1-3/Documents/Demo/from_gdwang/input_data/All_dataset.csv', feature_columns, label_column)

    type_mapping= {'發短球': 1, '長球': 2, '推撲球': 3, 
                '殺球': 4, '接殺防守': 5, '平球': 6, 
                '接不到': 11, '網前球': 7, '挑球': 8, 
                '切球': 9, '發長球': 10}
    
    data = pd.read_csv('/home/adsl-1-3/Documents/Demo/from_gdwang/input_data/merged_file.csv')
    label_input_data = []
    current_index = 0
    key_count = 0
    for index, row in data.iterrows():  
        mapped_type = type_mapping.get(row['type'], 0)
        if pd.isna(row['player_location_x']) or pd.isna(row['player_location_y']) :
            continue
        if row['win_reason'] == '落地致勝':
            input = [row['ball_round'],row['player_score']+row['opponent_score'],row['player_location_x'],row['player_location_y'],row['opponent_location_x'],row['opponent_location_x'],row['landing_x'],row['landing_y'],mapped_type,0]
            label_input_data.append(input) # player_location xy, opponent location xy, ball landing_location xy, ball_type
            current_index += 1
            key_count = 1

        elif pd.isna(row['win_reason']) and row['type']!='接不到':
            # if random.random() < 0.2 and key_count > 0:
                input = [row['ball_round'],row['player_score']+row['opponent_score'],row['player_location_x'],row['player_location_y'],row['opponent_location_x'],row['opponent_location_x'],row['landing_x'],row['landing_y'],mapped_type,1]
                label_input_data.append(input) # player_location xy, opponent location xy, ball landing_location xy, ball_type
                current_index += 1
                key_count -= 1
        elif row['win_reason'] == '對手掛網':
            pre_type = type_mapping.get(data.iloc[index-1]['type'], 0)
            # input = [data.iloc[index-1]['player_location_x'],data.iloc[index-1]['player_location_y'],data.iloc[index-1]['opponent_location_x'],data.iloc[index-1]['opponent_location_y'],data.iloc[index-1]['landing_x'],data.iloc[index-1]['landing_y'],pre_type,0]
            # label_input_data.append(input)
            # current_index += 1
            # key_count = 3
        else:
            continue

    columns = [
    'ball_round','rally',
    'player_location_x', 'player_location_y',
    'opponent_location_x', 'opponent_location_y',
    'landing_x', 'landing_y',
    'type', 'label'
    ]

    # Convert to DataFrame
    output_df = pd.DataFrame(label_input_data, columns=columns)

    # Save to CSV
    output_df.to_csv('/home/adsl-1-3/Documents/Demo/from_gdwang/input_data/Processed_dataset_merge2.csv', index=False)