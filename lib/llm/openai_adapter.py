# lib/llm/openai_adapter.py
from openai import OpenAI
import time
import json

class GPTClient:
    """
    Minimal adapter so existing code calling `client.generate(prompt)`
    or `client.generate_json(prompt)` will keep working.
    """

    def __init__(self, api_key=None, model="gpt-4o"):
        # If api_key is None, OpenAI client will read from env var OPENAI_API_KEY
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt, system=None, max_tokens=1024, temperature=0.0, **kwargs):
        """
        Returns plain text like GeminiClient.generate used to.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        # adapt to the likely shape
        try:
            return resp.choices[0].message["content"]
        except Exception:
            # fallback to string conversion
            return str(resp)

    def generate_json(self, prompt, **kwargs):
        """
        If older code expects a JSON object back, try to parse it.
        """
        txt = self.generate(prompt, **kwargs)
        try:
            return json.loads(txt)
        except Exception:
            # return raw text if not valid json
            return txt

    def generate_stream(self, prompt, **kwargs):
        """
        Simple streaming generator (yields chunks). Useful if any function used streaming.
        """
        messages = [{"role":"user","content":prompt}]
        stream = self.client.chat.completions.stream(
            model=self.model,
            messages=messages,
            **kwargs
        )
        for event in stream:
            # event may contain delta chunks
            # yield textual content if present
            try:
                for choice in getattr(event, "choices", []):
                    delta = choice.delta
                    if delta and "content" in delta:
                        yield delta["content"]
            except Exception:
                pass
