"""
llm.py: small wrapper around FREE TIER LLM providers.

Supported providers (both have a free tier and an OpenAI compatible API):
  gemini : GEMINI_API_KEY  (free key at https://aistudio.google.com/apikey)   <- default
  groq   : GROQ_API_KEY    (free key at https://console.groq.com/keys)

The key is read from a local .env file (never committed or submitted) or typed
into the app sidebar.

The agent keeps its conversation in a neutral format:
  {"role": "user", "content": str}
  {"role": "assistant", "text": str, "tool_calls": [{"id", "name", "args"}]}
  {"role": "tool", "results": [{"id", "name", "content"}]}
and this file converts it to the OpenAI chat format both providers accept.
"""
import json
import os
import time

PROVIDERS = {
    # name: (env var for key, default model, OpenAI compatible base URL)
    "gemini": ("GEMINI_API_KEY", "gemini-2.5-flash", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    "groq": ("GROQ_API_KEY", "openai/gpt-oss-120b", "https://api.groq.com/openai/v1"),
}


def load_dotenv(path=".env"):
    """Tiny .env reader so no extra package is needed."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def detect_provider():
    load_dotenv()
    for name, (env, _, _) in PROVIDERS.items():
        if os.environ.get(env):
            return name
    return "gemini"


class LLM:
    def __init__(self, provider, model=None, api_key=None, temperature=0.2, max_retries=4, max_tokens=2000):
        import openai

        load_dotenv()
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {provider}. Use one of {list(PROVIDERS)}.")
        env, default_model, base_url = PROVIDERS[provider]
        self.provider = provider
        self.model = model or os.environ.get("SCOUT_MODEL") or default_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_tokens = max_tokens  # cost cap: max output tokens per call
        key = api_key or os.environ.get(env)
        if not key:
            raise ValueError(f"No API key. Put {env}=... in a .env file or paste it in the sidebar.")
        self.client = openai.OpenAI(api_key=key, base_url=base_url)
        self._openai = openai

    def chat(self, system, messages, tools=None):
        """Returns {"text": str, "tool_calls": [{"id","name","args"}], "usage": {...}}"""
        msgs = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                d = {"role": "assistant", "content": m.get("text") or ""}
                if m.get("tool_calls"):
                    d["tool_calls"] = [{"id": tc["id"], "type": "function", "function": {
                        "name": tc["name"], "arguments": json.dumps(tc["args"])}} for tc in m["tool_calls"]]
                msgs.append(d)
            else:
                for r in m["results"]:
                    msgs.append({"role": "tool", "tool_call_id": r["id"], "name": r["name"],
                                 "content": r["content"]})
        kwargs = dict(model=self.model, messages=msgs, temperature=self.temperature, max_tokens=self.max_tokens)
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        if self.provider == "gemini" and getattr(self, "_reasoning_ok", True):
            kwargs["reasoning_effort"] = "low"  # keeps "thinking" tokens inside the output cap

        # Free tiers have per minute limits: wait and retry instead of crashing.
        for attempt in range(self.max_retries + 1):
            try:
                r = self.client.chat.completions.create(**kwargs)
                break
            except self._openai.RateLimitError:
                if attempt == self.max_retries:
                    raise
                time.sleep(10 * (attempt + 1))
            except self._openai.BadRequestError as e:
                if "reasoning" in str(e).lower() and "reasoning_effort" in kwargs:
                    kwargs.pop("reasoning_effort")  # model does not support it: retry without
                    self._reasoning_ok = False
                    continue
                raise

        msg = r.choices[0].message
        calls = []
        for i, tc in enumerate(msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append({"id": tc.id or f"call_{i}", "name": tc.function.name, "args": args})
        usage = {}
        if getattr(r, "usage", None):
            usage = {"input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens}
        return {"text": msg.content or "", "tool_calls": calls, "usage": usage}
