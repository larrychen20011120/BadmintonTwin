# NaiveRAG Evaluation

這個目錄包含 NaiveRAG Agent 的評估程式碼。

## 檔案結構

- `Agent/naive_rag.py` - NaiveRAG Agent 實現
- `Agent/eval_naive_rag.py` - 評估腳本
- `Experiments/naive_rag/data/` - 訓練資料庫（由 `baseline_prepare.py` 生成）
- `Experiments/naive_rag/results/` - 評估結果輸出目錄

## 準備工作

首先，確保已經運行了 `baseline_prepare.py` 來生成 NaiveRAG 資料庫：

```bash
python Preprocessing/baseline_prepare.py --naive-rag
```

這會在 `Experiments/naive_rag/data/` 目錄下生成每個選手的資料庫檔案（`*_db.pkl`）。

## 運行評估

### 基本使用

```bash
python Agent/eval_naive_rag.py
```

這會使用預設參數在 testing data 上評估 NaiveRAG Agent。

### 進階選項

```bash
python Agent/eval_naive_rag.py \
    --test_split Data/intermediate_data/train_test_split.json \
    --merged_csv Data/intermediate_data/merged.csv \
    --player_map Data/intermediate_data/player_index_map.json \
    --database_dir Experiments/naive_rag/data \
    --top_k 5 \
    --skip_first_n 0 \
    --output Experiments/naive_rag/results/naive_rag_eval_results.json
```

### 參數說明

- `--test_split`: Train/test split JSON 檔案路徑
- `--merged_csv`: 合併後的 CSV 資料路徑
- `--player_map`: 選手名稱到 ID 的對應檔案路徑
- `--database_dir`: NaiveRAG 資料庫目錄（預設: `Experiments/naive_rag/data`）
- `--top_k`: 檢索相似經驗的數量（預設: 5）
- `--skip_first_n`: 在每個 rally 中跳過前 N 個 shot（預設: 0）
- `--model_name`: LLM 模型名稱（預設: `Qwen/Qwen2.5-7B-Instruct`）
- `--port`: LLM 伺服器端口（預設: 16000）
- `--output`: 評估結果輸出檔案路徑

## 評估指標

評估會計算以下指標（與 BC 模型相同）：

1. **Land DTW**: 落點軌跡的 Dynamic Time Warping 距離
2. **Move DTW**: 移動軌跡的 Dynamic Time Warping 距離
3. **Shot CTC**: 球種預測的 CTC Loss
4. **Edit Distance**: 球種序列的編輯距離

## 相依需求

- 需要運行 LLM 伺服器（預設在 port 16000）
- 需要 `dtaidistance` 套件（應該在 conda 環境中）

## 輸出格式

評估結果會輸出為 JSON 檔案，包含每個選手的評分以及加權平均：

```json
{
  "Player Name": {
    "land_dtw": 0.1234,
    "move_dtw": 0.5678,
    "shot_ctc": 0.9012,
    "edit_dist": 0.3456,
    "num_rallies": 100,
    "num_shots": 500
  }
}
```
