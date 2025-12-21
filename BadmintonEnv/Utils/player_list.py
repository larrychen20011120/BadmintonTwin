import pandas as pd

# 假設你的 DataFrame 是這樣的
data = pd.read_csv('input_data/merged_file.csv')

# 計算每個人名出現次數並排序
player_count = data['player'].value_counts().reset_index()

# 重新命名欄位
player_count.columns = ['player', 'count']

# 顯示結果
print(player_count)

matched_rows = pd.DataFrame(columns=data.columns)

# 用 for 迴圈遍歷每一行
for i in range(len(data) - 1):
    # 確認第 i 行是 'Kento MOMOTA' 且下一行是 'Viktor AXELSEN'
    if (data['player'].iloc[i] == 'Kento MOMOTA' and 
        data['player'].iloc[i+1] == 'Viktor AXELSEN'):
        
        # 紀錄這兩行
        matched_rows = pd.concat([matched_rows, data.iloc[[i, i+1]]])

matched_rows.to_csv('input_data/row_K_V_data.csv', index=False)

data = data[((data['match_id'] == 13) | (data['match_id'] == 22)) & ((data['player'] == 'Kento MOMOTA') | (data['player'] == 'Viktor AXELSEN'))]
data.to_csv('input_data/K_V_data.csv', index=False)
