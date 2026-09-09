import copy
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from medpilot import prompts
from medpilot.client import ModelClient, OllamaClient, load_env
from medpilot.data import normalize
from medpilot.pipeline import DEFAULTS, execute, initialize
from medpilot.report import metrics, proportion, report


ROW = {"id": "123", "question": "Does treatment improve the outcome?", "contexts": ["A randomized study found improvement."],
       "gold_answer": "yes", "reference_explanation": "UNIQUE_SECRET_CONCLUSION"}
ANSWER = {"answer": "yes", "explanation": "The randomized study found improvement."}
JUDGE = {"reasoning_correctness": 5, "reasoning_completeness": 4, "hallucination": False,
         "consistency": True, "material_error": False, "issues": [], "assessment": "Supported by the supplied evidence."}


class FakeClient:
    def __init__(self, fail_judge=False):
        self.calls = []
        self.fail_judge = fail_judge

    def generate(self, system, prompt, schema, artifact):
        self.calls.append((prompt, schema))
        if schema is prompts.JUDGE_SCHEMA and self.fail_judge:
            raise RuntimeError("deliberate test failure")
        return {"parsed": copy.deepcopy(ANSWER if schema is prompts.ANSWER_SCHEMA else JUDGE),
                "model": "test-model", "usage": {}, "latency_seconds": 0}


class DataTests(unittest.TestCase):
    def test_solver_excludes_all_reference_fields(self):
        row = dict(ROW, reasoning_required_pred="no", final_decision="yes", LONG_ANSWER="secret")
        value = prompts.answer_input(row)
        self.assertEqual(set(json.loads(value)), {"question", "abstract"})
        self.assertNotIn("UNIQUE_SECRET_CONCLUSION", value)
        self.assertIn("UNIQUE_SECRET_CONCLUSION", prompts.judge_input(row, ANSWER))

    def test_conclusion_sections_and_duplicates_rejected(self):
        raw = {"x": {"QUESTION": "Q", "CONTEXTS": ["C"], "LABELS": ["CONCLUSIONS"],
                      "final_decision": "yes", "LONG_ANSWER": "L"}}
        with self.assertRaises(ValueError):
            normalize(raw)
        raw["x"]["LABELS"] = ["RESULTS"]
        raw["y"] = raw["x"].copy()
        with self.assertRaises(ValueError):
            normalize(raw)

    def test_output_validation_rejects_bad_scores_and_flags(self):
        for bad in [dict(JUDGE, reasoning_correctness=True), dict(JUDGE, reasoning_completeness=6),
                    dict(JUDGE, hallucination="false"), dict(JUDGE, material_error=True)]:
            with self.assertRaises(ValueError):
                prompts.validate(bad, prompts.JUDGE_SCHEMA)
        with self.assertRaises(ValueError):
            prompts.validate({"answer": "YES", "explanation": "x"}, prompts.ANSWER_SCHEMA)
        with self.assertRaises(ValueError):
            prompts.validate({"answer": "yes", "explanation": " "}, prompts.ANSWER_SCHEMA)

    def test_env_is_data_not_shell(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {}, clear=True):
            Path(root, ".env").write_text('OPENAI_API_KEY="$(touch /tmp/never_execute)"\n')
            load_env(root)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "$(touch /tmp/never_execute)")


class MetricTests(unittest.TestCase):
    def test_denominators_missing_judgments_and_false_reasoning(self):
        good = dict(ROW, answer_status="ok", prediction=ANSWER, judge_status="ok",
                    judgment=dict(JUDGE, material_error=True))
        unjudged = dict(ROW, id="2", answer_status="ok", prediction=ANSWER, judge_status="error")
        wrong = dict(ROW, id="3", answer_status="ok", prediction=dict(ANSWER, answer="no"),
                     judge_status="ok", judgment=JUDGE)
        missing = dict(ROW, id="4", answer_status="error")
        result = metrics([good, unjudged, wrong, missing])
        self.assertEqual(result["answer_accuracy"]["value_percent"], 50)
        self.assertEqual(result["false_reasoning_rate"]["denominator"], 1)
        self.assertEqual(result["false_reasoning_rate"]["value_percent"], 100)
        self.assertEqual(result["hallucination_rate"]["denominator"], 2)
        self.assertEqual(result["confusion_matrix"]["yes"]["invalid_or_missing"], 1)

    def test_zero_denominators_and_interval(self):
        self.assertIsNone(proportion(0, 0)["value_percent"])
        interval = proportion(5, 10)["ci95_percent"]
        self.assertEqual(interval, [23.66, 76.34])
        result = metrics([dict(ROW, answer_status="error")])
        self.assertIsNone(result["reasoning_correctness"]["mean"])
        self.assertIsNone(result["false_reasoning_rate"]["value_percent"])


class IntegrationTests(unittest.TestCase):
    def config(self):
        return dict(DEFAULTS, sample_size=1, output_dir="runs/test")

    def test_resume_skips_completed_calls_and_no_gold_leak(self):
        with tempfile.TemporaryDirectory() as root:
            client = FakeClient()
            folder = execute(self.config(), [ROW], root=root, client=client)
            execute(self.config(), [ROW], root=root, client=client)
            self.assertEqual(len(client.calls), 2)
            self.assertNotIn("UNIQUE_SECRET_CONCLUSION", client.calls[0][0])
            self.assertTrue((folder / "report.html").exists())
            self.assertTrue((folder / "human_review_TEMPLATE.csv").exists())
            self.assertEqual(report(folder)["status"], "COMPLETE")

    def test_failed_judge_preserves_answer_and_can_resume(self):
        with tempfile.TemporaryDirectory() as root:
            client = FakeClient(fail_judge=True)
            with self.assertRaises(RuntimeError):
                execute(self.config(), [ROW], root=root, client=client)
            folder = Path(root) / "runs/test"
            self.assertFalse((folder / ".run.lock").exists())
            self.assertEqual(report(folder)["status"], "PARTIAL / NOT RUN")
            client.fail_judge = False
            execute(self.config(), [ROW], root=root, client=client)
            self.assertEqual(len(client.calls), 2)
            execute(self.config(), [ROW], retry_failed=True, root=root, client=client)
            self.assertEqual(len(client.calls), 3)
            self.assertEqual(report(folder)["status"], "COMPLETE")

    def test_changed_config_or_mode_cannot_mix_runs(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.config()
            initialize(cfg, [ROW], "live", root)
            with self.assertRaises(ValueError):
                initialize(dict(cfg, seed=9), [ROW], "live", root)
            with self.assertRaises(ValueError):
                initialize(cfg, [ROW], "smoke", root)

    def test_html_escapes_model_content(self):
        with tempfile.TemporaryDirectory() as root:
            malicious = dict(ROW, question="<script>alert('x')</script>")
            folder = execute(self.config(), [malicious], mode="smoke", root=root)
            page = (folder / "report.html").read_text()
            self.assertNotIn("<script>", page)
            self.assertIn("&lt;script&gt;", page)
            self.assertIn("SYNTHETIC SMOKE TEST", page)


class HTTPTests(unittest.TestCase):
    def test_local_model_requires_no_key_and_records_usage(self):
        response = {"done": True, "done_reason": "stop", "model": "llama3.2:3b",
                    "message": {"content": json.dumps(ANSWER)}, "prompt_eval_count": 12, "eval_count": 24}
        cfg = dict(DEFAULTS, provider="ollama", model="llama3.2:3b")
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {}, clear=True):
            with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as request:
                result = OllamaClient(cfg).generate("s", "q", prompts.ANSWER_SCHEMA, Path(root)/"local")
            sent = request.call_args[0][0]
            self.assertTrue(sent.full_url.startswith("http://127.0.0.1:11434/"))
            self.assertNotIn("Authorization", sent.headers)
            self.assertEqual(json.loads(sent.data)["options"]["temperature"], 0)
            self.assertEqual(result["usage"]["total_tokens"], 36)
            self.assertEqual(result["parsed"], ANSWER)

    def test_local_truncation_is_not_scored(self):
        response = {"done": True, "done_reason": "length", "message": {"content": json.dumps(ANSWER)}}
        cfg = dict(DEFAULTS, provider="ollama", model="llama3.2:3b")
        with tempfile.TemporaryDirectory() as root:
            with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
                with self.assertRaises(ValueError):
                    OllamaClient(cfg).generate("s", "q", prompts.ANSWER_SCHEMA, Path(root)/"local")
            self.assertTrue((Path(root)/"local.attempt_1.response.json").exists())

    def test_response_parsing_and_secret_exclusion(self):
        response = {"id": "test", "status": "completed", "model": "gpt-5-test",
                    "output": [{"type": "reasoning", "summary": []},
                               {"type": "message", "content": [{"type": "output_text", "text": json.dumps(ANSWER)}]}],
                    "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}}
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"OPENAI_API_KEY": "fake_secret_123"}):
            with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as request:
                result = ModelClient(DEFAULTS).generate("system", "question", prompts.ANSWER_SCHEMA, Path(root)/"x.answer.call_1")
            self.assertEqual(result["parsed"], ANSWER)
            payload = json.loads(request.call_args[0][0].data)
            self.assertFalse(payload["store"])
            self.assertTrue(payload["text"]["format"]["strict"])
            for file in Path(root).glob("*.json"):
                self.assertNotIn("fake_secret_123", file.read_text())
            self.assertTrue((Path(root)/"x.answer.call_1.request.json").exists())

    def test_incomplete_response_is_saved_not_scored(self):
        response = {"status": "incomplete", "output": [], "usage": {"output_tokens": 4096}}
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"OPENAI_API_KEY": "fake"}):
            with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
                with self.assertRaises(ValueError):
                    ModelClient(DEFAULTS).generate("s", "q", prompts.ANSWER_SCHEMA, Path(root)/"x")
            self.assertTrue((Path(root)/"x.attempt_1.response.json").exists())

    def test_auth_failure_not_retried(self):
        error = urllib.error.HTTPError("https://api.openai.com", 401, "Unauthorized", {}, None)
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"OPENAI_API_KEY": "fake"}):
            with patch("urllib.request.urlopen", side_effect=error) as request:
                with self.assertRaises(RuntimeError):
                    ModelClient(DEFAULTS).generate("s", "q", prompts.ANSWER_SCHEMA, Path(root)/"x")
            self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
