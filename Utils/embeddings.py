import os
import re

import numpy as np
import yaml
from openai import OpenAI


DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "Scripts", "config.yaml")
)


try:
    from dotenv import load_dotenv

    load_dotenv(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, ".env")),
        override=False,
    )
except Exception:
    pass


def _resolve_env_value(value):
    if isinstance(value, dict):
        return {key: _resolve_env_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env_value(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        expression = match.group(1)
        if ":-" in expression:
            name, default = expression.split(":-", 1)
            return os.environ.get(name, default)
        return os.environ.get(expression, "")

    return re.sub(r"\$\{([^}]+)\}", replace, value)


def _load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config if isinstance(config, dict) else {}


def resolve_embedding_settings(config=None, config_path=None, provider=None):
    """Resolve the selected embedding provider without exposing its secret."""
    config_path = config_path or DEFAULT_CONFIG_PATH
    config = config if isinstance(config, dict) else _load_config(config_path)
    embedding_config = config.get("embedding", {})
    if not isinstance(embedding_config, dict):
        raise ValueError("embedding configuration must be a mapping")

    configured_provider = _resolve_env_value(embedding_config.get("active_provider"))
    selected_provider = (
        provider
        or os.environ.get("BADMINTON_EMBEDDING_PROVIDER")
        or configured_provider
        or "local"
    )
    selected_provider = str(selected_provider).strip().lower()

    providers = embedding_config.get("providers", {})
    provider_config = providers.get(selected_provider) if isinstance(providers, dict) else None
    if not isinstance(provider_config, dict):
        available = ", ".join(sorted(providers)) if isinstance(providers, dict) else "none"
        raise ValueError(
            f"Unknown embedding provider '{selected_provider}'. Available providers: {available or 'none'}"
        )
    provider_config = _resolve_env_value(provider_config)

    model_name = str(provider_config.get("model", "")).strip()
    if not model_name:
        raise ValueError(f"embedding.providers.{selected_provider}.model must not be empty")

    dimension = provider_config.get("dimension")
    if dimension is not None:
        try:
            dimension = int(dimension)
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding provider dimension must be a positive integer or null") from exc
        if dimension < 1:
            raise ValueError("embedding provider dimension must be a positive integer or null")

    default_headers = provider_config.get("default_headers", {})
    if not isinstance(default_headers, dict):
        raise ValueError("embedding provider default_headers must be a mapping")
    default_headers = {
        str(key): value
        for key, value in default_headers.items()
        if value is not None and value != ""
    }

    twin_config = config.get("twin", {})
    twin_config = twin_config if isinstance(twin_config, dict) else {}
    batch_size = twin_config.get("embedding_batch_size")
    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("twin.embedding_batch_size must be a positive integer or null")

    dtype = str(twin_config.get("embedding_dtype", "float32"))
    if dtype not in {"float16", "float32"}:
        raise ValueError("twin.embedding_dtype must be float16 or float32")

    settings = {
        "provider": selected_provider,
        "model_name": model_name,
        "dimension": dimension,
        "revision": provider_config.get("revision"),
        "device": provider_config.get("device", "auto"),
        "api_key": provider_config.get("api_key"),
        "base_url": provider_config.get("base_url"),
        "default_headers": default_headers,
        "batch_size": batch_size,
        "dtype": dtype,
        "normalize_embeddings": bool(twin_config.get("normalize_embeddings", True)),
    }
    if selected_provider == "openrouter":
        if not settings["api_key"]:
            raise ValueError("OPENROUTER_API_KEY is required for the openrouter embedding provider")
        if not settings["base_url"]:
            raise ValueError("OpenRouter embedding base_url must not be empty")
        if dimension is None:
            raise ValueError("OpenRouter embedding dimension must be configured")
    elif selected_provider == "local":
        device = str(settings["device"] or "auto").lower()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("local embedding device must be auto, cpu, or cuda")
        settings["device"] = device
    else:
        raise ValueError(f"Unsupported embedding provider: {selected_provider}")
    return settings


class LocalEmbeddingClient:
    def __init__(self, model_name, device="auto", revision=None, dimension=None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Local Twin embeddings require sentence-transformers. Run `uv sync --extra twin`."
            ) from exc

        model_kwargs = {}
        if revision:
            model_kwargs["revision"] = revision
        if device and str(device).lower() != "auto":
            model_kwargs["device"] = device
        self.model = SentenceTransformer(model_name, **model_kwargs)
        actual_dimension = int(self.model.get_sentence_embedding_dimension())
        if dimension is not None and actual_dimension != int(dimension):
            raise ValueError(
                f"Local embedding model produces {actual_dimension} dimensions, expected {dimension}."
            )
        self.dimension = actual_dimension

    def get_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False, batch_size=None):
        kwargs = {}
        if batch_size is not None:
            kwargs["batch_size"] = int(batch_size)
        return self.model.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            **kwargs,
        )


class OpenRouterEmbeddingClient:
    def __init__(self, model_name, api_key, base_url, dimension, default_headers=None,
                 client=None):
        self.model_name = str(model_name)
        self.dimension = int(dimension)
        if client is None:
            client_kwargs = {"api_key": api_key, "base_url": base_url}
            if default_headers:
                client_kwargs["default_headers"] = default_headers
            client = OpenAI(**client_kwargs)
        self.client = client

    def get_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False, batch_size=None):
        del show_progress_bar
        texts = list(texts)
        if not texts:
            result = np.zeros((0, self.dimension), dtype=np.float32)
            return result if convert_to_numpy else result.tolist()

        request_batch_size = int(batch_size or len(texts))
        if request_batch_size < 1:
            raise ValueError("Embedding batch size must be a positive integer.")
        rows = []
        for start in range(0, len(texts), request_batch_size):
            batch = texts[start:start + request_batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            indices = [item.index for item in ordered]
            if len(ordered) != len(batch) or indices != list(range(len(batch))):
                raise ValueError(
                    "OpenRouter returned incomplete or invalid embedding indices for "
                    f"{len(batch)} inputs."
                )
            rows.extend(item.embedding for item in ordered)

        embeddings = np.asarray(rows, dtype=np.float32)
        expected_shape = (len(texts), self.dimension)
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"OpenRouter embedding shape mismatch: received {embeddings.shape}, "
                f"expected {expected_shape}."
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("OpenRouter embeddings contain NaN or infinity.")
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("OpenRouter returned a zero-length embedding.")
            embeddings = embeddings / norms
        return embeddings if convert_to_numpy else embeddings.tolist()


def create_embedding_client(settings, client=None):
    provider = settings["provider"]
    if provider == "local":
        if client is not None:
            return client
        return LocalEmbeddingClient(
            settings["model_name"],
            device=settings.get("device", "auto"),
            revision=settings.get("revision"),
            dimension=settings.get("dimension"),
        )
    if provider == "openrouter":
        return OpenRouterEmbeddingClient(
            settings["model_name"],
            settings["api_key"],
            settings["base_url"],
            settings["dimension"],
            default_headers=settings.get("default_headers"),
            client=client,
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")
