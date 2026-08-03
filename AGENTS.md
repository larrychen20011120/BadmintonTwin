# BadmintonTwin - Agent Instructions

## Quick Start

**Environment Setup:**
```bash
uv sync --all-extras --frozen
```

**Start LLM Server (Required for Agent):**
```bash
uv sync --project Scripts/vllm --frozen
sh Scripts/serve_qwen.sh  # Serves Qwen2.5-7B on port 16000 via vLLM
```
The server uses an isolated CUDA 12.9 vLLM environment. Requires GPU with
~16GB VRAM (uses 92% utilization).

**Data Preprocessing (One-time):**
```bash
uv run python Preprocessing/data.py
uv run python Preprocessing/rally_split.py
uv run python Preprocessing/rally_clustering.py
```

## Running Experiments

**RallyNet physical-rally pipeline:**
```bash
uv run python Scripts/prepare_all_data.py --rallynet
uv run python Scripts/train_rallynet.py --stage all
uv run python Scripts/single_player_eval/eval_all_rallies.py --algorithm rallynet
```

**Physical-rally evaluation:**
```bash
uv run python Scripts/single_player_eval/eval_all_rallies.py --algorithm bc
uv run python Scripts/single_player_eval/eval_all_rallies.py --algorithm rallynet
uv run python Scripts/single_player_eval/eval_all_rallies.py --algorithm naive_rag
```

## Architecture

**Entry Points:**
- `Agent/bc.py` - Behavior Cloning agent (main baseline)
- `Agent/rallynet.py` - RallyNet SDE-based opponent model
- `Agent/rallynetv2.py` - Extended RallyNet
- `Agent/naive_rag.py` - RAG baseline using LLM
- `Agent/base_agent.py` - Abstract base class for all agents

**Key Directory Structure:**
```
Agent/              - Agent implementations (bc.py, rallynet.py, naive_rag.py, etc.)
Experiments/        - Experiment outputs organized by method
Preprocessing/      - Data and physical-rally preprocessing scripts
Utils/              - LLM wrapper (port 16000), badminton_config.py, helpers
Prompts/Agent/      - Jinja2 templates for LLM prompts
Schemas/Agent/      - JSON schemas for structured LLM output
Scripts/            - Training/evaluation scripts and config.yaml
Data/intermediate_data/ - Processed match and rally data
```

## Configuration

**Central Config:** `Scripts/config.yaml`
- Defines data, model, and result paths for all methods
- `rallynet_training` owns all RallyNet BC and SDE hyperparameters
- `rallynet` controls inference seed, sampling, and error threshold

## Environment & Actions

**State (input to agents):** `[opponent_shot_type, player_x, player_y, opponent_x, opponent_y, landing_x, landing_y]`
- Coordinates normalized internally: x=[-1, 1], y=[-1, 1]
- External bounds: x=[-177.5, 177.5], y=[-480, 480]

**Action:** `(shot_type, hit_position, landing_position, move_position, distance)`
- `shot_type`: 1-11 per `Utils/badminton_config.py` EN_SHOT2INDEX
- `landing_position`: x=[-177.5, 177.5], y=[0, 480]
- `move_position`: x=[-177.5, 177.5], y=[-480, 0]

**Shot Types:**
1=Serve short, 2=Clear, 3=Push Shot, 4=Smash, 5=Smash Defence, 6=Drive, 7=Net Shot, 8=Lob, 9=Drop, 10=Serve long, 11=Missed

## LLM Integration

- Local vLLM server expected on **port 16000**
- Default model: `Qwen/Qwen2.5-7B-Instruct`
- Client wrapper: `Utils/llm.py` - handles schema validation via JSON Schema
- Structured output enforced with retries (max 10 trials)

## Data Processing Notes

**Input datasets** (place in `Data/input_data/`):
- `shuttleset.csv`, `shuttleset22.csv`

**Pipeline outputs:**
- `Data/intermediate_data/merged.csv` - Combined dataset
- `Data/intermediate_data/style_shift.json` - Style shift classifications
- `Data/intermediate_data/transformed_matches.json` - Final train/test data

**Player mappings:** `Data/intermediate_data/player_index_map.json`

## Training Models

```bash
uv run python Scripts/train_bc.py
uv run python Scripts/train_rallynet.py --stage all
```
RallyNet weights are saved to `Experiments/rallynet/weight/rally_model.pth`.

## Important File References

- `Utils/badminton_config.py` - Shot type mappings, area groups
- `Utils/llm.py` - LLM client with schema validation
- `Scripts/config.yaml` - Centralized experiment configuration
- `Agent/base_agent.py` - Agent interface definition

## Gitignore Exclusions

Large artifacts excluded from git:
- `*.pth`, `*.trc`, `*.pkl`, `*.csv`, `*.png`
- `Experiments/**/results/`, `Experiments/**/data/`
- `__pycache__/`
