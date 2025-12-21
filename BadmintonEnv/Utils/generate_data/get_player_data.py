import pandas as pd
import random

if __name__ == "__main__":

    type_mapping= {'發短球': 1, '長球': 2, '推撲球': 3, 
                '殺球': 4, '接殺防守': 5, '平球': 6, 
                '接不到': 11, '網前球': 7, '挑球': 8, 
                '切球': 9, '發長球': 10}
    
    data = pd.read_csv('/home/adsl-1-3/Documents/Demo/from_gdwang/input_data/merged_file.csv')

    output_data = []
    player_name = 'Kento MOMOTA'

    for index, row in data.iterrows():  
        if row['player'] == player_name and row['ball_round']>1 and pd.isna(row['player_location_x']) == False and row['type']!='接不到':
            mapped_type = type_mapping.get(row['type'], 0)
            pre_mapped_type = type_mapping.get(data.iloc[index-1]['type'], 0)
        
            output = [
                # observation
                (row['player_location_x'])*177.5,
                (row['player_location_y']-1)*240,
                -(row['opponent_location_x'])*177.5,
                (row['opponent_location_y']+1)*240,
                (row['hit_x'])*177.5,
                (row['hit_y']-1)*240,
                pre_mapped_type,
                # action
                -(row['landing_x'])*177.5,
                (row['landing_y']+1)*240,
                (row['moving_x'])*177.5,
                (row['moving_y']-1)*240,
                mapped_type
            ]
            if len(output) == 11:
                print(output)
            output_data.append(output)

    columns = [
        'player_location_x',
        'player_location_y',
        'opponent_location_x', 
        'opponent_location_y',
        'hit_x',
        'hit_y',
        'opponent_type',
        'landing_x', 
        'landing_y',
        'moving_x',
        'moving_y',
        'type'
    ]
    output_df = pd.DataFrame(output_data, columns=columns)

    # Save to CSV
    output_df.to_csv('/home/adsl-1-3/Documents/Demo/from_gdwang/input_data/Player_dataset_merge.csv', index=False)