# BadmintonTwin

## Environment setup

Install [uv](https://docs.astral.sh/uv/) and synchronize the Python 3.12
environment from the committed lockfile:

```bash
uv sync --all-extras --frozen
```

The main environment uses the official PyTorch CUDA 12.8 wheels. Run project
commands through `uv run`, for example:

```bash
uv run python Scripts/train_bc.py --help
uv run python -m unittest discover -s tests
```

`environment.yaml` is retained temporarily as a legacy Conda fallback. New
dependency changes must be made in `pyproject.toml` and locked with `uv lock`.

## Serve LLM with vLLM

vLLM is isolated from the training environment because its compiled kernels
require a matching PyTorch/CUDA stack. Synchronize it once, then start Qwen on
port 16000:

```bash
uv sync --project Scripts/vllm --frozen
sh Scripts/serve_qwen.sh
```

The locked vLLM wheel uses CUDA 12.9 because the current stable release does
not publish a CUDA 12.8 wheel. The host NVIDIA driver must support CUDA 12.9,
and the configured 7B BF16 model requires substantially more than 4 GB VRAM.

## Preprocess the dataset

Create `Data/intermediate_data/merged.csv`, then build the independent
rally-profile controllability benchmark:

```bash
uv run python Preprocessing/data.py
uv run python Preprocessing/rally_split.py
uv run python Preprocessing/rally_clustering.py
```

`rally_split.py` keeps players with at least 50 usable training rallies across
at least three matches in the benchmark. It creates an atomic 80:20 split that
balances each eligible player's short, medium, and long rallies; sparse
players and every rally involving them remain training-only.
`rally_clustering.py` fits each eligible player's preprocessing and KMeans
model using training rallies only. The inertia elbow defines the center of a K
window with radius one; candidates in that window must contain at least five
rallies per cluster, then the highest silhouette score selects K. Match coverage
is stored as diagnostic metadata but does not exclude a cluster. The frozen
model assigns held-out rallies and
writes centroid-derived cluster instructions to
`Data/intermediate_data/rally_profiles.json`. Use `--skip-llm` to generate
deterministic fallback instructions without a running LLM server.

Both commands accept `--merged-csv` when the merged dataset is stored outside
the configured path, for example:

```bash
uv run python Preprocessing/rally_split.py --merged-csv Data/input_data/merged.csv
uv run python Preprocessing/rally_clustering.py --merged-csv Data/input_data/merged.csv --skip-llm
uv run --extra analysis python Scripts/plot_rally_profiles.py --merged-csv Data/input_data/merged.csv --clean
```

The plotting command writes one dashboard per clustered player to
`Experiments/rally_profiles/visualizations/`. Each dashboard shows train-fitted
PCA and UMAP projections of training rallies with 80% coverage ellipses,
alongside robust cluster signatures, landing and shot patterns, rally lengths,
and the saved tactical instructions. UMAP marks both its visual cluster center
and the nonlinear projection of the original KMeans center. Held-out data is
reserved for evaluation and is not drawn. UMAP is a visualization aid only;
clustering and validation remain in the original 11-dimensional feature space.

The existing `style_shift.py` and `split.py` commands remain available for the
legacy normal/shifted experiments.

## Baseline

BC, RallyNet, ShuttleNet, Random, NaiveRAG, TimeRAG, and Twin use the atomic rally
split directly. The split lists a rally under both participants, but data
preparation and evaluation deduplicate the physical key `(match_id, set,
rally)` before processing it. BC, RallyNet, ShuttleNet, and Random do not
consume tactical instructions. In instructed RAG runs,
each player uses their own rally-profile instruction.

RallyNet trains a successful-action model first, then freezes it while training
an SDE error model over complete rally histories. Its preparer records and
excludes malformed source rallies, writes the exact physical training keys and
split hash into metadata, and carries that provenance into the final checkpoint.

NaiveRAG, TimeRAG, and Twin use `two_stage_v1` action generation. Stage 1 independently
retrieves successful and missed evidence and predicts whether a legal return is
likely plus its shot type. Stage 2 filters the player database by that predicted
shot type before ranking `conditioned_top_k` cases, then generates only landing
and recovery positions. Their point distance uses player, hit, and opponent
coordinates plus `ball_distance_x/y`; changing this feature contract requires
rebuilding both RAG databases.
Twin first performs an additional aspect-generation pre-stage to construct its
semantic embedding query, then uses the same disjoint Stage 1 and Stage 2 action
contracts. A returned Twin decision therefore uses three LLM calls; a missed
decision skips action Stage 2 and uses two logical generation stages. As with
the other RAG agents, an invalid provider response may be retried within a stage.
Rally-profile instructions are controllability conditions assigned with the
train-fitted clustering pipeline, not causal features for forecasting an unseen
rally from its opening state.

```bash
# Build one training sample per valid shot in each physical training rally.
uv run python Scripts/prepare_all_data.py --bc --shuttlenet
uv run python Scripts/prepare_all_data.py --rallynet
uv run python Scripts/prepare_all_data.py --naive-rag --timerag
uv run python Scripts/prepare_all_data.py --twin
uv run python Scripts/prepare_twin_embeddings.py

uv run python Scripts/train_bc.py
uv run python Scripts/train_rallynet.py --stage all
uv run python Scripts/train_shuttlenet.py

# Simulate every physical test rally once, while retaining both player views.
uv run python Scripts/single_player_eval/eval_all_rallies.py \
  --algorithm bc --no-instruct
uv run python Scripts/single_player_eval/eval_all_rallies.py \
  --algorithm rallynet --no-instruct
uv run python Scripts/single_player_eval/eval_all_rallies.py \
  --algorithm shuttlenet --no-instruct
uv run python Scripts/single_player_eval/eval_all_rallies.py \
  --algorithm random --no-instruct --seed 42 --device cpu
uv run python Scripts/single_player_eval/eval_all_rallies.py \
  --algorithm pure_llm --workers 4
uv run python Scripts/single_player_eval/eval_all_rallies.py \
  --algorithm naive_rag --workers 4 --conditioned-top-k 10 --instruction-mode instruct
uv run python Scripts/single_player_eval/eval_all_rallies.py \
  --algorithm timerag --workers 4 --conditioned-top-k 10 --instruction-mode no_instruct
uv run python Scripts/single_player_eval/eval_all_rallies.py \
  --algorithm twin --workers 4 --conditioned-top-k 10 \
  --lambda-weight 0.5 --instruction-mode instruct

# Resume only missing, failed, or invalid rally JSON files.
uv run python Scripts/single_player_eval/eval_all_rallies.py \
  --algorithm naive_rag --workers 4 \
  --resume-run Experiments/naive_rag/trajectories/rally_instruct/run_001

# Score the latest completed trajectory run.
uv run python Scripts/single_player_eval/score.py \
  --run-root Experiments/bc/trajectories/rally_no_instruct \
  --output-dir Experiments/bc/results/rally_no_instruct/latest

# Score every completed run with rally-player sample mean and standard deviation.
uv run python Scripts/single_player_eval/score.py \
  --run-root Experiments/bc/trajectories/rally_no_instruct \
  --all \
  --output-dir Experiments/bc/results/rally_no_instruct/all
```

Each evaluation creates a `run_NNN/rallies/` directory and writes one JSON
object per rally simulation. Rollouts stop naturally when possible and are
capped at the number of valid ground-truth shots plus two. The all-rally
evaluator writes one shared trajectory per physical rally.
Random uses area-first coordinate sampling and a hard limit of
`max(ground-truth shots + 2, 5)`. Its first five actions sample landing and
movement areas 1-24 and cannot sample Missed; later actions may sample areas
1-32 and Missed. Random shot types are not corrected for serve situations:
rounds 1-5 sample types 1-10 and later rounds sample types 1-11.
`score.py` reports the six canonical metrics at the overall, per-player, and
per-player-cluster levels. LandDTW, MoveDTW, and ShotLCS are calculated after
each physical rally is split into its two player sequences. Without `--all`,
these rally-player values are averaged within the latest completed run. With
`--all`, every valid rally-player-run value contributes directly to the
reported mean and standard deviation; no best trajectory is selected. The JSD
metrics retain their scope-level aggregation, and RallyLenJSD uses the complete
physical rally length. The script writes `scores.json`, `scores.csv`, and the
terminal report in `table.txt`. Cluster-level scoring uses
`rally_profiles.json` only as evaluation metadata; BC and ShuttleNet do not
receive instructions as model input.

## Twin memory and embeddings

Twin data preparation has two independent stages. Canonical memory contains the
LLM annotations, states, and actions, but no model-specific embeddings.

Preview one rally before building the full memory database:

```bash
uv run python Scripts/preview_twin_memory.py \
  --player "Viktor AXELSEN" --match-id 99 --set 2 --rally 30 \
  --show-prompt --show-annotation
```

Add `--no-llm` for an offline fallback preview or `--output preview.json` to
save the complete prompt, annotation, and generated memories.

```bash
# Build canonical rally-level memory. This is the slow LLM annotation stage.
uv run python Scripts/prepare_all_data.py --twin

# Build embedding caches with the selected embedding provider from config.yaml.
uv run python Scripts/prepare_twin_embeddings.py
```

Set `BADMINTON_EMBEDDING_PROVIDER=local` to use the configured
`sentence-transformers` model. To use OpenRouter with
`openai/text-embedding-3-small`, set `OPENROUTER_API_KEY` and switch providers:

```bash
BADMINTON_EMBEDDING_PROVIDER=openrouter \
OPENROUTER_API_KEY=... \
uv run python Scripts/prepare_twin_embeddings.py
```

The default local model is `BAAI/bge-large-en-v1.5` with 1024 dimensions.
OpenRouter defaults to `openai/text-embedding-3-small` with 1536 dimensions and
can be changed with `OPENROUTER_EMBEDDING_MODEL` and
`OPENROUTER_EMBEDDING_DIMENSION`. Changing provider, model, dimension,
normalization, or dtype creates a separate cache without rebuilding canonical
memory. Use `--players PLAYER_NAME` for a subset or `--force` to rebuild an
otherwise valid cache. The embedding command always reads `Scripts/config.yaml`.

Twin evaluation requires both the canonical memory and the selected provider's
embedding cache. OpenRouter also performs one remote query-embedding request per
Twin prediction. Old Twin cache schemas and pickle files with inline embeddings
are unsupported; rebuild the embedding cache or canonical memory as applicable.

Case:
> In today’s match, I had no confidence. I just tried not to make mistakes and just tried to put the shuttle back into my opponent’s court
> https://bwfworldtour.bwfbadminton.com/news-single/2022/07/02/malaysia-open-close-to-finding-my-answer-says-momota


> "From the start it was tight, Axelsen's defense was difficult to penetrate. I tried to attack first but it turned out to make him more comfortable, he has a higher range," said Ginting.
> https://voi.id/en/sports/178029#google_vignette
