import argparse
import json
import os
import platform
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from . import prompts
from .client import ModelClient, OllamaClient, load_env
from .data import CHECKSUM, REVISION, digest, download, prepare, save_json
from .report import report

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = {"sample_size": 10, "seed": 42, "provider": "openai", "model": "gpt-5",
            "reasoning_effort": "low", "max_output_tokens": 4096, "timeout_seconds": 180,
            "max_attempts": 3, "judge_enabled": True, "output_dir": "runs/pubmedqa_gpt5_pilot"}


def config_read(path):
    overrides = json.loads(Path(path).read_text())
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown config fields: " + ", ".join(sorted(unknown)))
    cfg = dict(DEFAULTS, **overrides)
    for key in ["sample_size", "max_output_tokens", "timeout_seconds", "max_attempts"]:
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if type(cfg["seed"]) is not int or type(cfg["judge_enabled"]) is not bool:
        raise ValueError("Invalid seed or judge_enabled")
    if cfg["reasoning_effort"] not in {"minimal", "low", "medium", "high"}:
        raise ValueError("Unsupported reasoning_effort for GPT-5")
    if cfg["provider"] not in {"openai", "ollama"} or not isinstance(cfg["model"], str) or not cfg["model"].strip():
        raise ValueError("Configure an OpenAI or local Ollama model")
    if not isinstance(cfg["output_dir"], str) or not cfg["output_dir"].strip():
        raise ValueError("output_dir must be nonempty")
    return cfg


def folder_for(cfg, root=ROOT):
    path = Path(cfg["output_dir"])
    return path if path.is_absolute() else Path(root) / path


def initialize(cfg, sample, mode, root=ROOT):
    folder = folder_for(cfg, root)
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {"config": cfg, "sample": sample, "mode": mode, "dataset_sha256": CHECKSUM,
                "solver_prompt": prompts.ANSWER_SYSTEM, "judge_prompt": prompts.JUDGE_SYSTEM,
                "answer_schema": prompts.ANSWER_SCHEMA, "judge_schema": prompts.JUDGE_SCHEMA,
                "code_hashes": {p.name: digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    fingerprint = digest(json.dumps(protocol, sort_keys=True).encode())
    manifest_file = folder / "manifest.json"
    if manifest_file.exists():
        previous = json.loads(manifest_file.read_text())
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Run configuration/data/code changed. Choose a new output_dir to avoid mixing experiments.")
        if json.loads((folder / "sample.json").read_text()) != sample:
            raise ValueError("Frozen sample was modified")
    else:
        save_json(folder / "sample.json", sample)
        save_json(manifest_file, {"fingerprint": fingerprint, "config": cfg, "mode": mode,
                                 "created_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "python": platform.python_version(), "dataset_revision": REVISION,
                                 "dataset_sha256": CHECKSUM, "code_hashes": protocol["code_hashes"],
                                 "sample_ids": [r["id"] for r in sample],
                                 "sampling": "simple random development sample; not official test split"})
        save_json(folder / "prompts.json", {k: protocol[k] for k in ["solver_prompt", "judge_prompt", "answer_schema", "judge_schema"]})
    return folder


@contextmanager
def run_lock(folder):
    path = folder / ".run.lock"
    try:
        descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("Run is locked. Check no process is active before removing .run.lock") from None
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def execute(cfg, sample, mode="live", retry_failed=False, root=ROOT, client=None):
    folder = initialize(cfg, sample, mode, root)
    if client is None and mode == "live":
        try:
            client = OllamaClient(cfg) if cfg["provider"] == "ollama" else ModelClient(cfg)
        except ValueError:
            report(folder)
            raise
    with run_lock(folder):
        try:
            if mode == "live" and isinstance(client, OllamaClient):
                provenance = client.provenance()
                provenance_file = folder / "model_provenance.json"
                if provenance_file.exists() and json.loads(provenance_file.read_text()) != provenance:
                    raise ValueError("Local model/version changed. Use a new output_dir.")
                save_json(provenance_file, provenance)
            for index, row in enumerate(sample, 1):
                path = folder / "records" / (row["id"] + ".json")
                result = json.loads(path.read_text()) if path.exists() else {"id": row["id"], "mode": mode}
                if result.get("mode") != mode:
                    raise ValueError("Cannot mix synthetic and live results")
                for phase in ["answer", "judge"]:
                    if phase == "judge" and (not cfg["judge_enabled"] or result.get("answer_status") != "ok"):
                        continue
                    status = phase + "_status"
                    if result.get(status) == "ok" or (result.get(status) == "error" and not retry_failed):
                        continue
                    print("[%d/%d] %s %s" % (index, len(sample), row["id"], phase), flush=True)
                    try:
                        if mode == "smoke":
                            # Deliberately synthetic, independent of gold answer. Never a benchmark.
                            generated = {"parsed": ({"answer": ["yes", "no", "maybe"][index % 3],
                                          "explanation": "SYNTHETIC FIXTURE. No AI inference was performed."} if phase == "answer" else
                                          {"reasoning_correctness": 2, "reasoning_completeness": 1,
                                           "hallucination": False, "consistency": False, "material_error": True,
                                           "issues": ["Synthetic explanation contains no supporting evidence."],
                                           "assessment": "SYNTHETIC TEST GRADE; not a medical assessment."}),
                                         "model": "synthetic-fixture", "usage": {}, "latency_seconds": 0}
                        else:
                            history = result.get(phase + "_invocations", 0) + 1
                            result[phase + "_invocations"] = history
                            save_json(path, result)
                            generated = client.generate(
                                prompts.ANSWER_SYSTEM if phase == "answer" else prompts.JUDGE_SYSTEM,
                                prompts.answer_input(row) if phase == "answer" else prompts.judge_input(row, result["prediction"]),
                                prompts.ANSWER_SCHEMA if phase == "answer" else prompts.JUDGE_SCHEMA,
                                folder / "raw" / ("%s.%s.call_%d" % (row["id"], phase, history)))
                        schema = prompts.ANSWER_SCHEMA if phase == "answer" else prompts.JUDGE_SCHEMA
                        parsed = prompts.validate(generated["parsed"], schema)
                        result["prediction" if phase == "answer" else "judgment"] = parsed
                        result[phase + "_metadata"] = {k: v for k, v in generated.items() if k != "parsed"}
                        if phase == "answer":
                            result["model_returned"] = generated["model"]
                        result[status] = "ok"
                        result.pop(phase + "_error", None)
                        save_json(path, result)
                    except Exception as exc:
                        result[status] = "error"
                        # Avoid arbitrary exception payloads which could contain credentials.
                        result[phase + "_error"] = type(exc).__name__
                        save_json(path, result)
                        raise
        finally:
            report(folder)
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducible PubMedQA evaluation with OpenAI or local Ollama")
    parser.add_argument("command", choices=["download", "prepare", "doctor", "run", "report", "smoke"])
    parser.add_argument("--config", default=str(ROOT / "configs/pilot.json"))
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly rerun failed calls; may incur charges")
    args = parser.parse_args(argv)
    try:
        load_env(ROOT)
        cfg = config_read(args.config)
        if args.command == "download":
            print(download(ROOT))
        elif args.command == "doctor":
            print("Python:", platform.python_version())
            print("Dataset cached:", (ROOT / "data/raw/ori_pqal.json").exists())
            if cfg["provider"] == "openai":
                print("OPENAI_API_KEY:", "configured (not displayed)" if os.environ.get("OPENAI_API_KEY") else "MISSING")
            else:
                print("Provider: local Ollama; no API key or API charges")
            print("Model:", cfg["model"])
            print("Maximum logical calls:", cfg["sample_size"] * (2 if cfg["judge_enabled"] else 1))
            print("Each call has at most %d output tokens." % cfg["max_output_tokens"])
            print("Doctor does not call the API or verify account/model access.")
        elif args.command == "report":
            report(folder_for(cfg))
            print(folder_for(cfg) / "report.html")
        else:
            sample = prepare(ROOT, cfg["sample_size"], cfg["seed"])
            if args.command == "prepare":
                folder = initialize(cfg, sample, "live")
                report(folder)
                print("Prepared %d questions. %s" % (len(sample), folder / "report.html"))
            else:
                mode = "smoke" if args.command == "smoke" else "live"
                if mode == "smoke":
                    cfg["output_dir"] += "_SYNTHETIC_SMOKE"
                    cfg["model"] = "synthetic-fixture"
                folder = execute(cfg, sample, mode, args.retry_failed)
                print("Report:", folder / "report.html")
                if report(folder)["status"] != "COMPLETE":
                    print("Run is partial. Inspect errors; use --retry-failed only after fixing the cause.", file=sys.stderr)
                    return 1
    except (ValueError, RuntimeError, OSError) as exc:
        print("ERROR:", str(exc), file=sys.stderr)
        return 1
    return 0
