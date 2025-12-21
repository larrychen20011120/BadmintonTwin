import pandas as pd
import random

if __name__ == "__main__":
    df1 = pd.read_csv('/home/adsl-1-3/Documents/Demo/from_gdwang/input_data/All_dataset.csv')
    df2 = pd.read_csv('/home/adsl-1-3/Documents/Demo/from_gdwang/input_data/all_dataset(ShuttleSet22) (1).csv')

    # 垂直合併（將 df2 添加到 df1 下方）
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # 儲存合併後的檔案
    merged_df.to_csv('merged_file.csv', index=False)

    print("合併完成，儲存為 merged_file.csv")