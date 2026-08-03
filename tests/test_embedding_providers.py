import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from Agent.twin import TwinEmbeddingCache, embedding_model_key
from Utils.embeddings import OpenRouterEmbeddingClient, resolve_embedding_settings
from tests.test_twin_embedding_pipeline import FakeEmbeddingModel, make_database


def make_config(openrouter_api_key="test-secret"):
    return {
        "embedding": {
            "active_provider": "local",
            "providers": {
                "local": {
                    "model": "fake/local",
                    "device": "cpu",
                    "dimension": 3,
                },
                "openrouter": {
                    "model": "openai/text-embedding-3-small",
                    "api_key": openrouter_api_key,
                    "base_url": "https://openrouter.ai/api/v1",
                    "dimension": 1536,
                    "default_headers": {"X-Title": "BadmintonTwin"},
                },
            },
        },
        "twin": {
            "embedding_batch_size": 2,
            "embedding_dtype": "float32",
            "normalize_embeddings": True,
        },
    }


class FakeOpenAIClient:
    def __init__(self):
        self.requests = []
        self.embeddings = self

    def create(self, model, input, encoding_format):
        self.requests.append({
            "model": model,
            "input": list(input),
            "encoding_format": encoding_format,
        })
        data = []
        for index, text in enumerate(input):
            vector = [float(len(text)), float(ord(text[0])), 1.0]
            data.append(SimpleNamespace(index=index, embedding=vector))
        return SimpleNamespace(data=list(reversed(data)))


class EmbeddingProviderTest(unittest.TestCase):
    def test_resolves_openrouter_settings_from_environment(self):
        config = make_config(openrouter_api_key="${OPENROUTER_API_KEY}")
        with patch.dict(
            os.environ,
            {
                "BADMINTON_EMBEDDING_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "environment-secret",
            },
            clear=False,
        ):
            settings = resolve_embedding_settings(config=config)

        self.assertEqual(settings["provider"], "openrouter")
        self.assertEqual(settings["model_name"], "openai/text-embedding-3-small")
        self.assertEqual(settings["dimension"], 1536)
        self.assertEqual(settings["api_key"], "environment-secret")
        self.assertEqual(settings["batch_size"], 2)

    def test_openrouter_requires_api_key(self):
        config = make_config(openrouter_api_key="${OPENROUTER_API_KEY}")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                resolve_embedding_settings(config=config, provider="openrouter")

    def test_openrouter_batches_reorders_and_normalizes(self):
        fake_client = FakeOpenAIClient()
        client = OpenRouterEmbeddingClient(
            "openai/text-embedding-3-small",
            "test-secret",
            "https://openrouter.ai/api/v1",
            3,
            client=fake_client,
        )

        embeddings = client.encode(["aa", "bbb", "c"], batch_size=2)

        self.assertEqual(embeddings.shape, (3, 3))
        self.assertEqual(len(fake_client.requests), 2)
        self.assertEqual(fake_client.requests[0]["input"], ["aa", "bbb"])
        self.assertEqual(fake_client.requests[1]["input"], ["c"])
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), np.ones(3))
        expected_first = np.array([2.0, float(ord("a")), 1.0], dtype=np.float32)
        expected_first /= np.linalg.norm(expected_first)
        np.testing.assert_allclose(embeddings[0], expected_first)

    def test_provider_and_dimension_change_cache_key(self):
        local_key = embedding_model_key(
            "same/model", provider="local", embedding_dimension=3,
        )
        remote_key = embedding_model_key(
            "same/model", provider="openrouter", embedding_dimension=3,
        )
        other_dimension_key = embedding_model_key(
            "same/model", provider="openrouter", embedding_dimension=4,
        )
        self.assertNotEqual(local_key, remote_key)
        self.assertNotEqual(remote_key, other_dimension_key)

    def test_provider_cache_metadata_round_trip_has_no_secret(self):
        database = make_database()
        cache = TwinEmbeddingCache.build(
            database,
            FakeEmbeddingModel(),
            "openai/text-embedding-3-small",
            provider="openrouter",
            embedding_dimension=3,
        )

        with tempfile.TemporaryDirectory() as directory:
            _, metadata_path = cache.save(directory)
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            loaded = TwinEmbeddingCache.load(
                directory,
                database,
                "openai/text-embedding-3-small",
                provider="openrouter",
                embedding_dimension=3,
            )

            self.assertEqual(metadata["schema_version"], TwinEmbeddingCache.schema_version)
        self.assertEqual(metadata["provider"], "openrouter")
        self.assertEqual(metadata["configured_dimension"], 3)
        self.assertNotIn("api_key", metadata)
        self.assertEqual(loaded.embedding_dim, 3)


if __name__ == "__main__":
    unittest.main()
