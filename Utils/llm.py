from openai import OpenAI
import time
import re
import os
import copy
import json
import yaml

try:
    from dotenv import load_dotenv

    load_dotenv(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, ".env")),
        override=False,
    )
except Exception:
    pass

try:
    from json_repair import loads as repair_json_loads
except Exception:
    repair_json_loads = None

class LLM:

    def __init__(
        self,
        model_name=None,
        host=None,
        api_key=None,
        port=None,
        base_url=None,
        host_url=None,
        config_path=None,
        llm_type=None,
        provider=None,
        generation_params=None,
    ):
        settings = self.resolve_effective_settings(
            model_name=model_name,
            host=host,
            api_key=api_key,
            port=port,
            base_url=base_url,
            host_url=host_url,
            config_path=config_path,
            llm_type=llm_type,
            provider=provider,
            generation_params=generation_params,
            include_config=True,
        )

        self.config_path = settings["config_path"]
        self.config = settings["_config"]
        self.api_key = settings["api_key"]
        self.llm_type = settings.get("llm_type")
        self.model_name = settings["model_name"]
        self.provider = settings["provider"]
        self.generation_params = settings["generation_params"]

        endpoint_url = settings.get("base_url") or settings.get("host_url")

        client_kwargs = {"api_key": self.api_key}
        default_headers = settings.get("default_headers")
        if isinstance(default_headers, dict) and default_headers:
            client_kwargs["default_headers"] = default_headers
        if endpoint_url is not None:
            client_kwargs["base_url"] = endpoint_url
        elif self.api_key == "EMPTY":
            client_kwargs["base_url"] = f"http://{settings['host']}:{settings['port']}/v1"


        self.client = OpenAI(**client_kwargs)
        if self.api_key == "EMPTY":
            print("Local model loaded successfully.")

    @staticmethod
    def _deep_update(base, updates):
        merged = copy.deepcopy(base)
        if not isinstance(updates, dict):
            return merged
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged

    @classmethod
    def _resolve_env_value(cls, value):
        if isinstance(value, dict):
            return {key: cls._resolve_env_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._resolve_env_value(item) for item in value]
        if not isinstance(value, str):
            return value

        def _replace(match):
            expr = match.group(1)
            if ":-" in expr:
                name, default = expr.split(":-", 1)
                return os.environ.get(name, default)
            return os.environ.get(expr, "")

        return re.sub(r"\$\{([^}]+)\}", _replace, value)

    @staticmethod
    def _coalesce(*values):
        for value in values:
            if value is not None and value != "":
                return value
        return None

    @staticmethod
    def _default_config_path():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "Scripts", "config.yaml"))

    def _load_llm_config(self, config_path):
        if yaml is None or not os.path.exists(config_path):
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _resolve_client_config(self, model_name, host, api_key, port, base_url, host_url, llm_type=None, provider=None):
        llm_cfg = self.config.get("llm", {}) if isinstance(self.config, dict) else {}
        providers = llm_cfg.get("providers", {}) if isinstance(llm_cfg.get("providers", {}), dict) else {}

        configured_type = self._resolve_env_value(llm_cfg.get("active_type") or llm_cfg.get("default_type"))
        resolved_type = self._coalesce(
            llm_type,
            os.environ.get("BADMINTON_LLM_TYPE"),
            configured_type,
            "qwen",
        )
        resolved_type = str(resolved_type).lower()

        configured_provider = self._resolve_env_value(llm_cfg.get("active_provider") or llm_cfg.get("default_provider"))
        resolved_provider = self._coalesce(
            provider,
            os.environ.get("BADMINTON_LLM_PROVIDER"),
            configured_provider,
            "local",
        )
        resolved_provider = str(resolved_provider).lower()

        provider_cfg = providers.get(resolved_provider)
        if not isinstance(provider_cfg, dict):
            available = ", ".join(sorted(providers.keys())) or "none"
            raise ValueError(f"Unknown LLM provider '{resolved_provider}'. Available providers: {available}")
        provider_cfg = self._resolve_env_value(provider_cfg)

        models = provider_cfg.get("models", {}) if isinstance(provider_cfg.get("models", {}), dict) else {}
        configured_model_name = models.get(resolved_type)
        if model_name is None and configured_model_name is None:
            available = ", ".join(sorted(models.keys())) or "none"
            raise ValueError(
                f"Unknown LLM type '{resolved_type}' for provider '{resolved_provider}'. "
                f"Available types: {available}"
            )

        resolved_port = self._coalesce(port, provider_cfg.get("port"), 16000)
        try:
            resolved_port = int(resolved_port)
        except (TypeError, ValueError):
            resolved_port = 16000

        resolved_api_key = self._coalesce(api_key, provider_cfg.get("api_key"), "EMPTY")
        provider_headers = provider_cfg.get("default_headers", {})
        default_headers = self._deep_update(
            {},
            provider_headers if isinstance(provider_headers, dict) else {},
        )
        default_headers = {
            key: value for key, value in default_headers.items()
            if value is not None and value != ""
        }

        return {
            "llm_type": resolved_type,
            "provider": resolved_provider,
            "model_name": self._coalesce(
                model_name,
                configured_model_name,
                "Qwen/Qwen2.5-7B-Instruct",
            ),
            "api_key": resolved_api_key,
            "host": self._coalesce(host, provider_cfg.get("host"), "localhost"),
            "port": resolved_port,
            "base_url": self._coalesce(base_url, provider_cfg.get("base_url")),
            "host_url": self._coalesce(host_url, provider_cfg.get("host_url")),
            "default_headers": default_headers,
        }

    @classmethod
    def resolve_effective_settings(
        cls,
        model_name=None,
        host=None,
        api_key=None,
        port=None,
        base_url=None,
        host_url=None,
        config_path=None,
        llm_type=None,
        provider=None,
        generation_params=None,
        include_config=False,
    ):
        config_path = config_path or cls._default_config_path()
        instance = cls.__new__(cls)
        instance.config_path = config_path
        instance.config = instance._load_llm_config(config_path)
        resolved = instance._resolve_client_config(
            model_name=model_name,
            host=host,
            api_key=api_key,
            port=port,
            base_url=base_url,
            host_url=host_url,
            llm_type=llm_type,
            provider=provider,
        )

        generation = instance._resolve_generation_params(
            llm_type=resolved.get("llm_type"),
            provider=resolved["provider"],
            overrides=generation_params,
        )

        llm_gen_cfg = instance.config.get("llm_generation", {}) if isinstance(instance.config, dict) else {}
        deterministic_cfg = llm_gen_cfg.get("deterministic", {}) if isinstance(llm_gen_cfg, dict) else {}
        settings = {
            **resolved,
            "config_path": config_path,
            "generation_params": generation,
            "deterministic_active": bool(deterministic_cfg.get("active", False)) if isinstance(deterministic_cfg, dict) else False,
        }
        if include_config:
            settings["_config"] = instance.config
        return settings

    def _resolve_generation_params(self, llm_type=None, provider=None, overrides=None):
        config = self.config
        params = dict()

        llm_gen_cfg = config.get("llm_generation", {}) if isinstance(config, dict) else {}
        deterministic_cfg = llm_gen_cfg.get("deterministic", {}) if isinstance(llm_gen_cfg.get("deterministic", {}), dict) else {}
        provider_overrides = llm_gen_cfg.get("provider_overrides", {}) if isinstance(llm_gen_cfg.get("provider_overrides", {}), dict) else {}

        def _model_params():
            by_type = llm_gen_cfg.get("by_type", {})
            if not isinstance(by_type, dict):
                return {}
            selected = by_type.get(str(llm_type).lower()) if llm_type is not None else None
            return selected if isinstance(selected, dict) else {}

        if isinstance(deterministic_cfg, dict) and deterministic_cfg.get("active", False):
            params = self._deep_update(params, deterministic_cfg.get("params", {}))
        else:
            params = self._deep_update(params, _model_params())

        provider_name = str(provider or "").lower()
        params = self._deep_update(params, provider_overrides.get(provider_name, {}))

        return self._deep_update(params, overrides)

    def get_verified_response(self, schema=None, system_prompt="", user_prompt="", max_trials=10,
                              validator=None):
        
        trials = 0
        last_raw_text = ""
        last_usage = {}

        while trials < max_trials:

            llm_response_str, usage = self.get_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            last_raw_text = llm_response_str
            if usage:
                last_usage = usage

            data = self._extract_json(llm_response_str)
            if data is not None:
                if validator is not None:
                    try:
                        if not validator(data):
                            print(user_prompt)
                            print("=" * 30)
                            print(llm_response_str)
                            print("=" * 30)
                            print(data)
                            print("Invalid structured response; retrying LLM generation")
                            trials += 1
                            continue
                    except (TypeError, ValueError):
                        print(user_prompt)
                        print("=" * 30)
                        print(llm_response_str)
                        print("=" * 30)
                        print(data)
                        print("TypeError or ValueError during validation; retrying LLM generation")
                        trials += 1
                        continue
                # print(user_prompt)
                # print("=" * 30)
                # print(llm_response_str)
                # print("=" * 30)

                return data, llm_response_str, last_usage
            else:
                print(user_prompt)
                print("=" * 30)
                print(llm_response_str)
                print("❌ No parseable JSON found ❌")

            trials += 1
        return None, last_raw_text, last_usage

    @staticmethod
    def _extract_json(raw_text):
        """Parse a JSON-only LLM response with light repair for common mistakes."""
        if not raw_text or not isinstance(raw_text, str):
            return None

        text = raw_text.strip()

        fenced = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if fenced:
            text = fenced.group(1).strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        if repair_json_loads is not None:
            try:
                return repair_json_loads(text)
            except Exception:
                pass

        repaired_text = re.sub(r"([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', text)
        repaired_text = re.sub(r",\s*([}\]])", r"\1", repaired_text)
        repaired_text = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', repaired_text)

        try:
            return json.loads(repaired_text)
        except Exception:
            return None

    def get_response(self, system_prompt="", user_prompt=""):

        start_time = time.time()

        response_text = ""
        usage = {}

        if len(system_prompt) != 0:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        else:
            messages = [
                {"role": "user", "content": user_prompt}
            ]

        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            **self.generation_params,
        }
        # print(request_kwargs)
        response = self.client.chat.completions.create(**request_kwargs)
        response_text = response.choices[0].message.content

        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        end_time = time.time()

        return response_text, usage
