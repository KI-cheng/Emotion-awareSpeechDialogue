import requests
from config.config import API_KEY


class LLMClient:
    def __init__(self, api_key=None, endpoint=None, model="ln-gpt35-turbo"):
        self.api_key = api_key or API_KEY
        """用gpt4o的话，改成ln-gpt40"""
        self.model = model
        self.endpoint = endpoint or f"https://genai-jp.openai.azure.com/openai/deployments/{model}/chat/completions?api-version=2023-03-15-preview"
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        self.conversation_history = []

    def get_response(self, ask, emo, temperature=1, top_p=0.95, max_tokens=400, keep_history=False):
        prompt = """You are an empathetic psychological comfort expert. Next, I will give you what the user said and 
        the corresponding emotions. Please respond to the user from the perspective of comforting the user (within 30 
        words). User: {}, Emotion: {}""".format(ask, emo)
        if keep_history and self.conversation_history:
            messages = self.conversation_history + [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "system", "content": prompt}]
        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']

            if keep_history:
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": content})

            return content

        except requests.RequestException as e:
            raise SystemExit(f"Failed to make the request. Error: {e}")
        except KeyError:
            raise ValueError("Unexpected response format: Could not find expected fields in the response.")

    def clear_history(self):
        self.conversation_history = []

    def batch_process(self, prompts_list, **kwargs):
        results = []
        for prompt in prompts_list:
            try:
                response = self.get_response(prompt, **kwargs)
                results.append(response)
            except Exception as e:
                results.append(f"Error: {e}")
        return results
