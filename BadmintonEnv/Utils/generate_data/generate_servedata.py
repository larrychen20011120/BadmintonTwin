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
    
    data = pd.read_csv('input_data/merged_file.csv')
    label_input_data = []

    for index, row in data.iterrows():  
        mapped_type = type_mapping.get(row['type'], 0)
        if row['ball_round'] != 1:
            continue

        if pd.isna(row['win_reason']) and row['type']!='接不到':
            input = [row['player'],row['player_location_x'],row['player_location_y'],row['opponent_location_x'],row['opponent_location_x'],mapped_type,
                     row['landing_x'],row['landing_y'],row['moving_x'],row['moving_y']]
            
            flag = 0
            for i in input:
                if pd.isna(i):flag = 1
            if flag == 1:continue

            label_input_data.append(input)

    columns = [
    'player',
    'player_location_x', 'player_location_y',
    'opponent_location_x', 'opponent_location_y',
    'type',
    'landing_x', 'landing_y',
    'moving_x','moving_y'
    ]

    # Convert to DataFrame
    output_df = pd.DataFrame(label_input_data, columns=columns)

    for index, row in output_df.iterrows():  
        for i in row:
            if pd.isna(i):
                print('11111111111111111111111111111111111111')

    # Save to CSV
    output_df.to_csv('input_data/Processed_dataset_merge3.csv', index=False)