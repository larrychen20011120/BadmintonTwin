# NaiveRAG Evaluation

這個目錄包含 two-stage NaiveRAG Agent 的評估資料與輸出。

## 檔案結構

- `Agent/naive_rag.py` - NaiveRAG Agent 實現
- `Agent/eval_naive_rag.py` - 評估腳本
- `Experiments/naive_rag/data/rally/` - 每位選手的 rally-level 訓練資料庫
- `Experiments/naive_rag/results/` - 評估結果輸出目錄

## 準備工作

首先，使用統一資料準備入口生成 NaiveRAG 資料庫：

```bash
uv run python Scripts/prepare_all_data.py --naive-rag
```

這會依 `(match_id, set, rally)` 精確選取 train rallies，並在
`Experiments/naive_rag/data/rally/` 生成每個選手的 `*_db.pkl`。

## 運行評估

### 基本使用

```bash
uv run python Scripts/single_player_eval/eval_all_rallies.py \
  --algorithm naive_rag --workers 4 --conditioned-top-k 10 \
  --instruction-mode no_instruct
```

### 進階選項

每個 physical rally 會立即原子寫入一個 JSON；中斷後可用
`--resume-run <run_NNN>` 只補跑缺失或失敗的 rallies。

### Two-stage 流程

Stage 1 依 config 分別檢索 normal/miss evidence，LLM 只決定
miss/return 與球種。若為 return，Stage 2 先按預測球種過濾 DB，再以
`--conditioned-top-k` 控制 placement evidence 數量。每個非 missed shot
會呼叫 LLM 兩次；missed shot 只呼叫一次。

使用的 prompts：

- `Prompts/New/naive_rag/stage1_norm.jinja`
- `Prompts/New/naive_rag/stage1_instruct.jinja`
- `Prompts/New/naive_rag/stage2_norm.jinja`
- `Prompts/New/naive_rag/stage2_instruct.jinja`

## 評估指標

評估會計算以下指標（與 BC 模型相同）：

1. **Land DTW**: 落點軌跡的 Dynamic Time Warping 距離
2. **Move DTW**: 移動軌跡的 Dynamic Time Warping 距離
3. **Shot CTC**: 球種預測的 CTC Loss
4. **Edit Distance**: 球種序列的編輯距離

## 相依需求

- 需要運行 LLM 伺服器（預設在 port 16000）
- 需要 `dtaidistance` 套件（由 `uv sync` 安裝）

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
