from openai import OpenAI
import time
import re
import os
import copy
from jsonschema import validate, ValidationError
from json_repair import loads as repair_json_loads

try:
    import yaml
except Exception:
    yaml = None

class LLM:

    _DEFAULT_GENERATION = {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 2048,
        "extra_body": {
            "repetition_penalty": 1.05,
        },
    }

    def __init__(
        self,
        model_name,
        host="140.113.213.17",
        api_key="EMPTY",
        port=16000,
        base_url=None,
        config_path=None,
        generation_profile=None,
        generation_params=None,
    ):

        self.api_key = api_key
        self.model_name = model_name
        self.config_path = (
            config_path
            if config_path is not None
            else os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "Scripts", "config.yaml"))
        )
        self.generation_profile = generation_profile
        self.generation_params = self._resolve_generation_params(
            model_name=model_name,
            profile_name=generation_profile,
            overrides=generation_params,
        )

        if api_key == "EMPTY":

            self.client = OpenAI(
                api_key=api_key, 
                base_url=f"http://{host}:{port}/v1"
            )
            
            print("Local model loaded successfully.")

        else:
            client_kwargs = {"api_key": api_key}
            if base_url is not None:
                client_kwargs["base_url"] = base_url
            self.client = OpenAI(**client_kwargs)

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

    @staticmethod
    def _normalize_model_name(model_name):
        if not isinstance(model_name, str):
            return model_name
        return model_name.split("/")[-1]

    def _load_llm_config(self, config_path):
        if yaml is None or not os.path.exists(config_path):
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _resolve_generation_params(self, model_name, profile_name=None, overrides=None):
        config = self._load_llm_config(self.config_path)
        params = copy.deepcopy(self._DEFAULT_GENERATION)

        llm_gen_cfg = config.get("llm_generation", {}) if isinstance(config, dict) else {}
        params = self._deep_update(params, llm_gen_cfg.get("default", {}))

        by_model = llm_gen_cfg.get("by_model", {})
        if isinstance(by_model, dict):
            selected_profile = by_model.get(model_name)
            if selected_profile is None:
                selected_profile = by_model.get(self._normalize_model_name(model_name))
            if profile_name is not None:
                selected_profile = by_model.get(profile_name, selected_profile)
            params = self._deep_update(params, selected_profile)

        return self._deep_update(params, overrides)

    def get_verified_response(self, schema, system_prompt="", user_prompt="", max_trials=10):
        
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
                try:
                    validate(instance=data, schema=schema)
                    return data, llm_response_str, last_usage
                except ValidationError as e:
                    print(user_prompt)
                    print("=" * 30)
                    print(llm_response_str)
                    print(f"❌ Structure Error {e.message} ❌")

            trials += 1
        
        return None, last_raw_text, last_usage

    @staticmethod
    def _extract_json(raw_text):
        """Extract and repair JSON from LLM text."""
        if not raw_text or not isinstance(raw_text, str):
            return None

        def _repair(candidate):
            candidate = candidate.strip()
            if not candidate:
                return None
            try:
                return repair_json_loads(candidate)
            except Exception:
                return None

        def _object_candidates(text):
            candidates = []
            start = None
            depth = 0
            in_string = False
            escape_next = False

            for index, char in enumerate(text):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == '{':
                    if depth == 0:
                        start = index
                    depth += 1
                elif char == '}' and depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        candidates.append(text[start:index + 1])
                        start = None

            return candidates

        candidates = []
        fenced_blocks = [
            match.group(1).strip()
            for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
        ]
        candidates.extend(fenced_blocks)
        for block in fenced_blocks:
            candidates.extend(reversed(_object_candidates(block)))

        candidates.extend(reversed(_object_candidates(raw_text)))

        first_brace = raw_text.find('{')
        last_brace = raw_text.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            candidates.append(raw_text[first_brace:last_brace + 1])
        candidates.append(raw_text.strip())

        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            parsed = _repair(candidate)
            if parsed is not None:
                return parsed

        print("=" * 30)
        print(raw_text)
        print("❌ Json Decode Error ❌")
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

        if self.api_key != "EMPTY":
            request_kwargs.pop("extra_body", None)

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
