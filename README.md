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
- create the merged file `Data/input/merged.csv` of shuttleset and shuttleset22
- create the `Data/offline_data/style_shift.json` record the style shifted matches and normal matches
- apply train-test split and data aggregation on `Data/offline_data/style_shift.json` to generate the `Data/offline_data/transformed_matches.json`
```bash
uv run python Preprocessing/data.py
uv run python Preprocessing/style_shift.py
uv run python Preprocessing/split.py
```

## Baseline

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
# Build canonical memory. This is the slow LLM annotation stage.
uv run python Scripts/prepare_all_data.py --norm --shifted --twin

# Build embedding caches with the selected embedding provider from config.yaml.
uv run python Scripts/prepare_twin_embeddings.py --norm --shifted
```

Set `BADMINTON_EMBEDDING_PROVIDER=local` to use the configured
`sentence-transformers` model. To use OpenRouter with
`openai/text-embedding-3-small`, set `OPENROUTER_API_KEY` and switch providers:

```bash
BADMINTON_EMBEDDING_PROVIDER=openrouter \
OPENROUTER_API_KEY=... \
uv run python Scripts/prepare_twin_embeddings.py --norm --shifted
```

The model defaults to `openai/text-embedding-3-small` with 1536 dimensions and
can be changed with `OPENROUTER_EMBEDDING_MODEL` and
`OPENROUTER_EMBEDDING_DIMENSION`. Changing provider, model, dimension,
normalization, or dtype creates a separate cache without rebuilding canonical
memory. Use `--players PLAYER_NAME` for a subset or `--force` to rebuild an
otherwise valid cache. The embedding command always reads `Scripts/config.yaml`.

Twin evaluation requires both the canonical memory and the selected provider's
embedding cache. OpenRouter also performs one remote query-embedding request per
Twin prediction. Cache schema 1 and old Twin pickle files with inline embeddings
are unsupported; rebuild the embedding cache or canonical memory as applicable.

Case:
> In today’s match, I had no confidence. I just tried not to make mistakes and just tried to put the shuttle back into my opponent’s court
> https://bwfworldtour.bwfbadminton.com/news-single/2022/07/02/malaysia-open-close-to-finding-my-answer-says-momota


> "From the start it was tight, Axelsen's defense was difficult to penetrate. I tried to attack first but it turned out to make him more comfortable, he has a higher range," said Ginting.
> https://voi.id/en/sports/178029#google_vignette
