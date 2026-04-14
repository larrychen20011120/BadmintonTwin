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

