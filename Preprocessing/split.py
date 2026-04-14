import json
import random
import os

def split_dataset(data, seed=42):
    random.seed(seed)

    train_dict = {}
    test_dict = {}

    for player in data["normal"]:
        name = player["name"]
        match_ids = sorted(player["match_ids"])

        if len(match_ids) >= 4:
            test_id = random.sample(match_ids, 1)[0]
            train_ids = [m for m in match_ids if m != test_id]

            train_dict[name] = train_ids
            test_dict[name] = [test_id]

        else:
            # 只有 train
            train_dict[name] = match_ids

    return {
        "train": train_dict,
        "test": test_dict
    }


with open(os.path.join("Data", "offline_data", "style_shift.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

result = split_dataset(data, seed=42)

with open(os.path.join("Data", "offline_data", "train_test_split.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("Done")