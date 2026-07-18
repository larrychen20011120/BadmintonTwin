# BadmintonTwin

## Serve LLM with vLLM

we offer three local LLM to justify the framework tranferability (Qwen / Mistral / LLaMA)
```
sh serve_qwen.sh
```

## Preprocess the dataset
- create the merged file `Data/input/merged.csv` of shuttleset and shuttleset22
- create the `Data/offline_data/style_shift.json` record the style shifted matches and normal matches
- apply train-test split and data aggregation on `Data/offline_data/style_shift.json` to generate the `Data/offline_data/transformed_matches.json`
```
python data.py
python style_shift.py
python split.py
```

## Baseline

## Twin memory and embeddings

Twin data preparation has two independent stages. Canonical memory contains the
LLM annotations, states, and actions, but no model-specific embeddings.

```bash
# Build canonical memory. This is the slow LLM annotation stage.
python Scripts/prepare_all_data.py --norm --shifted --twin

# Build embedding caches with twin.embedding_model from Scripts/config.yaml.
python Scripts/prepare_twin_embeddings.py --norm --shifted
```

Changing `twin.embedding_model` and rerunning the second command creates a new
model-specific cache without rebuilding memory or overwriting other embedding
models. Use `--players PLAYER_NAME` for a subset or `--force` to rebuild an
otherwise valid cache. The embedding command always reads `Scripts/config.yaml`.

Twin evaluation requires both the canonical memory and the configured model's
embedding cache. Old Twin pickle files with inline embeddings are unsupported;
rebuild canonical memory before using this pipeline.

Case:
> In today’s match, I had no confidence. I just tried not to make mistakes and just tried to put the shuttle back into my opponent’s court
> https://bwfworldtour.bwfbadminton.com/news-single/2022/07/02/malaysia-open-close-to-finding-my-answer-says-momota


> "From the start it was tight, Axelsen's defense was difficult to penetrate. I tried to attack first but it turned out to make him more comfortable, he has a higher range," said Ginting.
> https://voi.id/en/sports/178029#google_vignette
