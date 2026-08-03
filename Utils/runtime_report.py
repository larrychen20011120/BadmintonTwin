import json
import os


SENSITIVE_KEYS = ("api_key", "access_token", "bearer_token", "secret", "password")
PRIVATE_LOCATION_KEYS = ("host", "base_url", "host_url", "url")


def mask_sensitive(value):
    if value is None or value == "":
        return "<unset>"
    text = str(value)
    if text == "EMPTY":
        return "EMPTY"
    if len(text) <= 8:
        return "<masked>"
    return f"{text[:3]}...{text[-3:]}"


def format_value(value, key=""):
    key_lower = key.lower()
    if any(part in key_lower for part in SENSITIVE_KEYS):
        return mask_sensitive(value)
    if any(key_lower.endswith(part) for part in PRIVATE_LOCATION_KEYS):
        return mask_sensitive(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None or value == "":
        return "<unset>"
    return str(value)


def print_runtime_box(title, rows, width=96):
    """Print key runtime parameters in a terminal-friendly box."""
    flat_rows = []
    for key, value in rows:
        if value is None:
            continue
        flat_rows.append((str(key), format_value(value, str(key))))

    if not flat_rows:
        return

    inner_width = max(width - 2, len(title) + 4)
    title_text = f" {title} "
    left = max((inner_width - len(title_text)) // 2, 0)
    right = max(inner_width - len(title_text) - left, 0)

    print("╭" + "─" * left + title_text + "─" * right + "╮")
    key_width = min(max(len(key) for key, _ in flat_rows), 32)
    for key, value in flat_rows:
        line = f"{key:<{key_width}}  {value}"
        if len(line) > inner_width:
            line = line[:inner_width - 3] + "..."
        print("│" + line.ljust(inner_width) + "│")
    print("╰" + "─" * inner_width + "╯")


def load_project_config(config_path=None):
    import yaml

    if config_path is None:
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "Scripts", "config.yaml"))
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f), config_path


def mask_secret(value):
    if value is None or value == "":
        return value
    if value == "EMPTY":
        return value
    value = str(value)
    if len(value) <= 8:
        return "<set>"
    return f"{value[:4]}...{value[-4:]}"


def llm_rows(settings, prefix="llm"):
    generation = settings.get("generation_params", {}) if isinstance(settings, dict) else {}
    return [
        (f"{prefix}.type", settings.get("llm_type")),
        (f"{prefix}.provider", settings.get("provider")),
        (f"{prefix}.model_name", settings.get("model_name")),
        (f"{prefix}.api_key", mask_secret(settings.get("api_key"))),
        (f"{prefix}.host", settings.get("host")),
        (f"{prefix}.port", settings.get("port")),
        (f"{prefix}.base_url", settings.get("base_url")),
        (f"{prefix}.host_url", settings.get("host_url")),
        (f"{prefix}.deterministic.active", settings.get("deterministic_active")),
        (f"{prefix}.generation.temperature", generation.get("temperature")),
        (f"{prefix}.generation.top_p", generation.get("top_p")),
        (f"{prefix}.generation.max_tokens", generation.get("max_tokens")),
        (f"{prefix}.generation.extra_body", generation.get("extra_body")),
    ]


def algorithm_runtime_rows(config, algorithm, **effective):
    rows = [
        ("algorithm", algorithm),
        ("experiment", effective.get("experiment")),
        ("player", effective.get("player")),
        ("device", effective.get("device")),
        ("skip_first_n", effective.get("skip_first_n")),
    ]

    retrieval = config.get("retrieval", {}) if isinstance(config, dict) else {}
    if algorithm in {"naive_rag", "timerag", "twin", "twin2"}:
        if algorithm in {"naive_rag", "timerag", "twin"}:
            stage1 = retrieval.get("stage1", {})
            stage2 = retrieval.get("stage2", {})
            rows.append(("retrieval.architecture", effective.get(
                "retrieval_mode", (
                    "twin_aspect_two_stage_v1"
                    if algorithm == "twin"
                    else retrieval.get("architecture", "two_stage_v1")
                )
            )))
            rows.append(("retrieval.stage1.normal_top_k", effective.get(
                "normal_top_k", stage1.get("normal_top_k")
            )))
            rows.append(("retrieval.stage1.miss_top_k", effective.get(
                "miss_top_k", stage1.get("miss_top_k")
            )))
            rows.append(("retrieval.stage2.conditioned_top_k", effective.get(
                "top_k", stage2.get("conditioned_top_k")
            )))
        else:
            rows.append(("retrieval.mode", effective.get("retrieval_mode", retrieval.get("mode", "classic"))))
            rows.append(("retrieval.classic_top_k", effective.get("top_k", retrieval.get("classic_top_k", retrieval.get("top_k")))))
    if algorithm == "timerag":
        rows.append(("timerag.window_size", effective.get("window_size", config.get("timerag", {}).get("window_size"))))
    elif algorithm == "twin":
        twin_cfg = config.get("twin", {})
        rows.append(("twin.lambda_weight", effective.get("lambda_weight", twin_cfg.get("lambda_weight"))))
        rows.append(("embedding.provider", effective.get("embedding_provider")))
        rows.append(("embedding.model", effective.get("embedding_model")))
        rows.append(("embedding.dimension", effective.get("embedding_dimension")))
        rows.append(("embedding.revision", effective.get("embedding_revision")))
        rows.append(("embedding.device", effective.get("embedding_device")))
        rows.append(("twin.embedding_batch_size", effective.get("embedding_batch_size", twin_cfg.get("embedding_batch_size"))))
        rows.append(("twin.embedding_dtype", effective.get("embedding_dtype", twin_cfg.get("embedding_dtype"))))
        rows.append(("twin.normalize_embeddings", effective.get("normalize_embeddings", twin_cfg.get("normalize_embeddings"))))
    elif algorithm == "twin2":
        rows.append(("twin2.lambda_weight", effective.get("lambda_weight", config.get("twin2", {}).get("lambda_weight"))))
    elif algorithm == "bc":
        bc_cfg = config.get("bc_training", {})
        rows.extend([
            ("bc.epochs", effective.get("epochs", bc_cfg.get("epochs"))),
            ("bc.batch_size", effective.get("batch_size", bc_cfg.get("batch_size"))),
            ("bc.learning_rate", effective.get("learning_rate", bc_cfg.get("learning_rate"))),
            ("bc.distribution", effective.get("distribution", bc_cfg.get("distribution"))),
            ("bc.gmm_num", effective.get("gmm_num", bc_cfg.get("gmm_num"))),
        ])
    elif algorithm == "shuttlenet":
        shuttlenet_cfg = config.get("shuttlenet_training", {})
        rows.extend([
            ("shuttlenet.epochs", effective.get("epochs", shuttlenet_cfg.get("epochs"))),
            ("shuttlenet.batch_size", effective.get("batch_size", shuttlenet_cfg.get("batch_size"))),
            ("shuttlenet.learning_rate", effective.get("learning_rate", shuttlenet_cfg.get("learning_rate"))),
            ("shuttlenet.encode_length", effective.get("encode_length", shuttlenet_cfg.get("encode_length"))),
            ("shuttlenet.max_ball_round", effective.get("max_ball_round", shuttlenet_cfg.get("max_ball_round"))),
        ])
    elif algorithm == "random":
        rows.append(("random.seed", effective.get("seed")))

    return rows
