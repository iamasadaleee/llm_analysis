"""Auditable stdlib HTTP adapter; raw requests/responses are saved per attempt."""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from .data import save_json
from .prompts import validate


class OllamaClient:
    """Local-only Ollama backend. No credentials or external inference service."""
    URL = "http://127.0.0.1:11434"

    def __init__(self, config):
        self.config = config

    def provenance(self):
        with urllib.request.urlopen(self.URL + "/api/tags", timeout=10) as response:
            tags = json.loads(response.read())
        matches = [m for m in tags.get("models", []) if m["name"] == self.config["model"]]
        if len(matches) != 1:
            raise ValueError("Local model is not installed. Run: ollama pull " + self.config["model"])
        with urllib.request.urlopen(self.URL + "/api/version", timeout=10) as response:
            version = json.loads(response.read())
        return {"provider": "ollama", "server_version": version.get("version"),
                "model": matches[0]["name"], "digest": matches[0]["digest"],
                "details": matches[0].get("details", {})}

    def generate(self, system, prompt, schema, artifact):
        cfg = self.config
        payload = {"model": cfg["model"], "stream": False, "format": schema,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": prompt}],
                   "options": {"temperature": 0, "seed": cfg["seed"], "num_ctx": 8192,
                               "num_predict": cfg["max_output_tokens"]}, "keep_alive": "10m"}
        save_json(str(artifact) + ".request.json", payload)
        started = time.monotonic()
        request = urllib.request.Request(self.URL + "/api/chat", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=cfg["timeout_seconds"]) as response:
                raw = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            save_json(str(artifact) + ".attempt_1.error.json", {"http_status": exc.code})
            raise RuntimeError("Ollama HTTP %s; inspect server logs and installed model" % exc.code) from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError("Local Ollama connection failed. Start ollama serve and check the timeout.") from None
        save_json(str(artifact) + ".attempt_1.response.json", raw)
        if not raw.get("done") or raw.get("done_reason") == "length":
            raise ValueError("Local response is incomplete; inspect raw output and token limit")
        # No repair using the gold answer; malformed outputs remain auditable failures.
        parsed = validate(json.loads(raw.get("message", {}).get("content", "")), schema)
        input_tokens, output_tokens = raw.get("prompt_eval_count", 0), raw.get("eval_count", 0)
        return {"parsed": parsed, "model": raw.get("model", cfg["model"]),
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                          "total_tokens": input_tokens + output_tokens},
                "latency_seconds": time.monotonic() - started, "attempts": 1}


def load_env(root):
    path = Path(root) / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(".env must contain NAME=value lines")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key == "OPENAI_API_KEY" and value and not os.environ.get(key):
            os.environ[key] = value


class ModelClient:
    def __init__(self, config):
        self.config = config
        if config["provider"] != "openai":
            raise ValueError("This pilot supports OpenAI; do not silently substitute models")
        self.key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not self.key:
            raise ValueError("OPENAI_API_KEY is missing. Copy .env.example to .env and enter the key locally.")

    def generate(self, system, prompt, schema, artifact):
        cfg = self.config
        payload = {"model": cfg["model"], "instructions": system, "input": prompt,
                   "store": False, "max_output_tokens": cfg["max_output_tokens"],
                   "reasoning": {"effort": cfg["reasoning_effort"]},
                   "text": {"format": {"type": "json_schema", "name": "evaluation_output",
                                       "strict": True, "schema": schema}}}
        artifact = Path(artifact)
        save_json(str(artifact) + ".request.json", payload)
        for attempt in range(1, cfg["max_attempts"] + 1):
            prefix = str(artifact) + ".attempt_%d" % attempt
            start = time.monotonic()
            request = urllib.request.Request("https://api.openai.com/v1/responses",
                                             data=json.dumps(payload).encode(), method="POST",
                                             headers={"Authorization": "Bearer " + self.key,
                                                      "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=cfg["timeout_seconds"]) as response:
                    raw = json.loads(response.read())
                elapsed = time.monotonic() - start
                save_json(prefix + ".response.json", raw)
                if raw.get("status") != "completed":
                    raise ValueError("Response incomplete/refused; inspect raw response and token budget")
                parts = [part["text"] for output in raw.get("output", [])
                         if output.get("type") == "message" for part in output.get("content", [])
                         if part.get("type") == "output_text"]
                parsed = validate(json.loads("".join(parts)), schema)
                return {"parsed": parsed, "model": raw.get("model", cfg["model"]),
                        "response_id": raw.get("id"), "usage": raw.get("usage", {}),
                        "latency_seconds": elapsed, "attempts": attempt}
            except urllib.error.HTTPError as exc:
                # Do not log request headers, API credentials, or arbitrary server error body.
                save_json(prefix + ".error.json", {"http_status": exc.code})
                if exc.code not in {429, 500, 502, 503, 504} or attempt == cfg["max_attempts"]:
                    raise RuntimeError("OpenAI HTTP %s; check account access, billing, model and request settings" % exc.code) from None
                time.sleep(min(2 ** attempt, 8))
            except (urllib.error.URLError, TimeoutError) as exc:
                save_json(prefix + ".error.json", {"error": type(exc).__name__})
                # A timeout can have been billed. Never retry unknown-completion requests automatically.
                raise RuntimeError("Network request failed; response completion is unknown. Inspect logs before resuming.") from None
        raise RuntimeError("API attempts exhausted")
