# BadmintonTwin - Agent Instructions

## Quick Start

**Environment Setup:**
```bash
conda env create -f environment.yaml
conda activate badminton
```

**Start LLM Server (Required for Agent):**
```bash
sh Scripts/serve_qwen.sh  # Serves Qwen2.5-7B on port 16000
```

**Preprocess Data (One-time):**
```bash
python Preprocessing/data.py      # Merges shuttleset + shuttleset22
python Preprocessing/style_shift.py  # Identifies style-shifted matches
python Preprocessing/split.py     # Train/test split
```

## Architecture

**Entry Points:**
- `Agent/twin.py` - Main LLM-based badminton agent (TwoStageLLMAgent)
- `BadmintonEnv/Environment.py` - Gym-like environment for rally simulation
- `BadmintonEnv/Agent/RallyNet.py` - BC model for opponent behavior cloning

**Key Directory Structure:**
```
Agent/          - Agent implementations (twin.py, bc.py, rallynet.py, naive_rag.py)
BadmintonEnv/   - Environment + RallyNet BC models
Preprocessing/  - Data pipeline scripts
Utils/          - LLM wrapper (port 16000), config, helpers
Prompts/        - Jinja2 templates for LLM prompts
Schemas/        - JSON schemas for structured LLM output
Data/input_data/    - Raw shuttleset CSVs
Data/offline_data/  - Processed match data
Data/output_data/   - Simulation outputs
```

## Running the Agent

```bash
python Agent/twin.py \
  --player "Kento MOMOTA" \
  --opponent "Viktor AXELSEN" \
  --n_rallies 50 \
  --filename my_output_game \
  --situation "選手現在腳受傷，移動能力不如往常"
```

**Arguments:**
- `--player` - Player agent name (target)
- `--opponent` - Opponent name or 'ALL' for BC baseline
- `--constraint/--no-constraint` - Enable physics constraints
- `--n_rallies` - Number of rallies to simulate
- `--situation` - Chinese context for the agent (injuries, conditions)

**Output:** `Data/output_data/{filename}.csv` with rally trajectory

## Environment & Actions

**State (8 dims):** `[opponent_shot_type, player_x, player_y, opponent_x, opponent_y, landing_x, landing_y]`
- Coordinates normalized to [-1, 1] internally
- External bounds: x=[-177.5, 177.5], y=[-480, 480]

**Action:** `(shot_type, hit_position, landing_position, move_position, distance)`
- `shot_type`: 1-11 (Serve short, Clear, Push Shot, Smash, Smash Defence, Drive, Net Shot, Lob, Drop, Serve long, Missed)
- `landing_position`: x=[-177.5, 177.5], y=[0, 480]
- `move_position`: x=[-177.5, 177.5], y=[-480, 0]

## Data Processing Notes

- Input datasets: `shuttleset.csv`, `shuttleset22.csv` (place in `Data/input_data/`)
- Output of pipeline: `transformed_matches.json` (train/test split)
- Shot type mapping in `Utils/badminton_config.py`: EN_SHOT2INDEX, area_groups

## LLM Integration

- Local vLLM server expected on port 16000
- Config: `BadmintonGPT/openai_two_stage.yaml` (referenced but not in repo)
- Structured output validated against JSON schemas
- Default model: Qwen/Qwen2.5-7B-Instruct

## Important Paths

- Weights: `BadmintonEnv/Agent/weight/Ralleynet/gen_e_1000.trc`
- Miss-hit model: `BadmintonEnv/Agent/BC_model/model_weights_merge6.pth`
- Target players: `input_data/target_players_ids.pkl`
- Hyperparameters: `BadmintonEnv/Agent/hyperparameters.pkl`

## Gitignore Exclusions

Large artifacts excluded from git:
- `*.pth`, `*.trc`, `*.pkl`, `*.csv`, `*.png`
- `Shot_Evaluation/`, `Tactic_Evaluation/`, `Test_preprocess/`
- `Result/`, `player/`, `*.ipynb`
