# gemini_adapter.py
import os, json, re
from types import SimpleNamespace
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# 2. Define the permissive safety settings
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

def _is_probably_json(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))

def _join_openai_messages(messages):
    system_parts = [m["content"] for m in (messages or []) if m.get("role") == "system"]
    convo_parts = []
    for m in messages or []:
        r, c = m.get("role"), m.get("content", "")
        if r == "user":
            convo_parts.append(f"User:\n{c}")
        elif r == "assistant":
            convo_parts.append(f"Assistant:\n{c}")
    system = "\n".join(system_parts)
    prompt = (system + "\n\n" if system else "") + "\n\n".join(convo_parts)
    return prompt

class _ChatCompletions:
    def __init__(self, default_model="gemini-2.5-flash", generation_config=None):
        self.default_model = default_model
        self.default_cfg = {"candidate_count": 1}
        if generation_config:
            self.default_cfg.update(generation_config)

    def create(self, model=None, messages=None, temperature=0.0, **kwargs):
        # Detect whether caller likely expects JSON
        expects_json = any(
            _is_probably_json(m.get("content", ""))
            for m in (messages or [])
            if m.get("role") in ("system", "user")
        )

        prompt = _join_openai_messages(messages)
        mdl = model or self.default_model
        gmodel = genai.GenerativeModel(mdl)

        gen_cfg = dict(self.default_cfg)
        gen_cfg["temperature"] = temperature
        for k in ("top_p", "top_k", "max_output_tokens"):
            if k in kwargs and kwargs[k] is not None:
                gen_cfg[k] = kwargs[k]
        if expects_json:
            gen_cfg["response_mime_type"] = "application/json"

        resp = gmodel.generate_content(prompt, generation_config=gen_cfg,safety_settings=SAFETY_SETTINGS)

        # Normalize to OpenAI shape
        text = (getattr(resp, "text", None) or "").strip()
        if not text and getattr(resp, "candidates", None):
            parts = getattr(resp.candidates[0].content, "parts", [])
            if parts:
                # prefer text parts; fallback to inline JSON string
                for p in parts:
                    if hasattr(p, "text") and p.text:
                        text = p.text.strip()
                        break
                if not text and hasattr(parts[0], "inline_data"):
                    try:
                        text = parts[0].inline_data.data.decode("utf-8").strip()
                    except Exception:
                        text = ""

        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

class GeminiClient:
    """Drop-in for OpenAI: client.chat.completions.create(...)"""
    def __init__(self, api_key=None, model="gemini-1.5-flash", generation_config=None):
        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Missing API key. Set GOOGLE_API_KEY or OPENAI_API_KEY.")
        genai.configure(api_key=key)
        self.chat = SimpleNamespace(completions=_ChatCompletions(model, generation_config))