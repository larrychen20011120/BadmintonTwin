from openai import OpenAI
import json
import time
from jsonschema import validate, ValidationError

class LLM:

    def __init__(self, model_name, api_key="EMPTY", port=16000):

        self.api_key = api_key
        self.model_name = model_name

        if api_key == "EMPTY":

            self.client = OpenAI(
                api_key=api_key, 
                base_url=f"http://localhost:{port}/v1"
            )
            
            print("Local model loaded successfully.")

        else:
            self.client = OpenAI(api_key=api_key)

    def get_verified_response(self, schema, system_prompt="", user_prompt="", max_trials=10):
        
        trials = 0

        while trials < max_trials:

            llm_response_str = self.get_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            try:
                data = json.loads(llm_response_str)
                validate(instance=data, schema=schema)
                break
            except json.JSONDecodeError:
                print(user_prompt)
                print("=" * 30)
                print(llm_response_str)
                print("❌ Json Decode Error ❌")
            except ValidationError as e:
                print(user_prompt)
                print("=" * 30)
                print(llm_response_str)
                print(f"❌ Structure Error {e.message} ❌")

            trials += 1
        
        if trials == max_trials:
            return None
        
        return data



    def get_response(self, system_prompt="", user_prompt=""):

        start_time = time.time()

        response_text = ""

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
                temperature=0.2,
                top_p=0.1,
                max_tokens=1024,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
        response_text = response.choices[0].message.content

        end_time = time.time()
        print(f"Response generated in {(end_time - start_time):.4f} seconds.")
        print(user_prompt)
        print("=" * 30)
        print(response_text)

        return response_text