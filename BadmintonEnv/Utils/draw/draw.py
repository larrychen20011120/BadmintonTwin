import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 文件
def read_csv(file_path):
    return pd.read_csv(file_path)

# 繪製並存成 PNG 散點圖
def plot_scatter(data, x_col, y_col, output_path):
    filtered_data = data#data[data['type_number'] == 'Clear']
    plt.figure(figsize=(8, 6))
    plt.scatter(filtered_data[x_col], filtered_data[y_col], alpha=0.5, edgecolors='k')
    plt.title(f'Scatter Plot of {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path, dpi=300)  # ✅ 儲存為 PNG
    plt.close()  # ✅ 關閉圖表以節省資源
    print(f'散點圖已存為 {output_path}')

if __name__ == "__main__":
    # 指定 CSV 文件路徑與輸出圖片路徑
    csv_file = '/home/adsl-1-3/Documents/Demo/from_gdwang/output_data/output_game_4.csv'
    output_png = '/home/adsl-1-3/Documents/Demo/from_gdwang/output_data/scatter_plot.png'
    
    # 讀取數據
    df = read_csv(csv_file)
    
    # 指定要繪製的兩列
    x_column = 'landing_x'
    y_column = 'landing_y'
    
    # 繪製並存成 PNG
    plot_scatter(df, x_column, y_column, output_png)
