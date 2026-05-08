from openai import OpenAI
import json
import time
import re
from jsonschema import validate, ValidationError

class LLM:

    def __init__(self, model_name, host="140.113.213.17", api_key="EMPTY", port=16000):

        self.api_key = api_key
        self.model_name = model_name

        if api_key == "EMPTY":

            self.client = OpenAI(
                api_key=api_key, 
                base_url=f"http://{host}:{port}/v1"
            )
            
            print("Local model loaded successfully.")

        else:
            self.client = OpenAI(api_key=api_key)

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
        """Extract a valid JSON object from raw text that may contain markdown fences or surrounding commentary.

        Tries: (1) direct parse, (2) strip ```json fences and parse, (3) find first { ... } pair.
        """
        if not raw_text or not isinstance(raw_text, str):
            return None
            
        # Pre-clean: remove isolated quotes at the end that don't have a matching opening quote
        # This handles cases like: {"key": "value"}"
        cleaned_text = raw_text.strip()
        while cleaned_text.endswith('"') and not cleaned_text.startswith('"'):
            # Check if this is an isolated quote (not part of a proper JSON string)
            # Count quotes before the last one
            quote_count = cleaned_text[:-1].count('"')
            if quote_count % 2 == 0:  # Even number of quotes means the last one is isolated
                cleaned_text = cleaned_text[:-1].strip()
            else:
                break
        
        # Attempt 1: direct json.loads on cleaned text
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass

        # Attempt 2: strip markdown code fences
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence line
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1:]
            # Remove closing fence
            if cleaned.strip().endswith("```"):
                cleaned = cleaned.strip()[:-3].strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        # Attempt 3: find JSON object boundaries more precisely
        # Look for the first { that starts a JSON object
        start = -1
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(raw_text):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    if brace_count == 0:
                        start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start != -1:
                        # Found a complete JSON object
                        json_str = raw_text[start:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            # Try to fix common issues
                            try:
                                # Remove trailing commas before } or ]
                                fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                                # Replace single quotes with double quotes (but not in strings)
                                # This is a simple heuristic
                                return json.loads(fixed)
                            except json.JSONDecodeError:
                                pass
                        start = -1
        
        # Attempt 4: try to extract anything that looks like a JSON object
        # Find the last occurrence of a complete JSON-like structure
        json_pattern = r'\{[^{}]*\}'
        for match in re.finditer(json_pattern, raw_text):
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
        
        # Attempt 5: more aggressive pattern matching
        # Find the text between the first { and last }
        first_brace = raw_text.find('{')
        last_brace = raw_text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_candidate = raw_text[first_brace:last_brace+1]
            # Try various fixes
            fixes = [
                lambda x: x,  # Original
                lambda x: re.sub(r',(\s*[}\]])', r'\1', x),  # Remove trailing commas
                lambda x: x.replace("'", '"'),  # Replace single quotes
                lambda x: re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', x),  # Add quotes to keys
            ]
            for fix in fixes:
                try:
                    return json.loads(fix(json_candidate))
                except json.JSONDecodeError:
                    continue

        # All attempts failed
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

        # use api-based agent
        if self.api_key == "EMPTY":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                top_p=0.8,
                max_tokens=2048,
                extra_body={ "repetition_penalty": 1.05, },
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
        response_text = response.choices[0].message.content

        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        end_time = time.time()

        return response_text, usage
