import csv
import html
import json
import math
from collections import Counter
from pathlib import Path
from .data import save_json


def proportion(successes, total):
    if not total:
        return {"value_percent": None, "numerator": 0, "denominator": 0, "ci95_percent": None}
    p, z = successes / total, 1.959963984540054
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return {"value_percent": round(100*p, 2), "numerator": successes, "denominator": total,
            "ci95_percent": [round(100*max(0, center-half), 2), round(100*min(1, center+half), 2)]}


def mean_score(judged, key):
    values = [r["judgment"][key] for r in judged]
    return {"mean": round(sum(values)/len(values), 3) if values else None,
            "n": len(values), "scale": "1-5", "distribution": dict(Counter(values))}


def metrics(rows):
    answered = [r for r in rows if r.get("answer_status") == "ok"]
    correct = [r for r in answered if r["prediction"]["answer"] == r["gold_answer"]]
    judged = [r for r in answered if r.get("judge_status") == "ok"]
    correct_judged = [r for r in judged if r["prediction"]["answer"] == r["gold_answer"]]
    confusion = {gold: {pred: 0 for pred in ["yes", "no", "maybe", "invalid_or_missing"]}
                 for gold in ["yes", "no", "maybe"]}
    for r in rows:
        pred = r["prediction"]["answer"] if r.get("answer_status") == "ok" else "invalid_or_missing"
        confusion[r["gold_answer"]][pred] += 1
    recalls = []
    for label, counts in confusion.items():
        if sum(counts.values()):
            recalls.append(counts[label] / sum(counts.values()))
    result = {
        "n_selected": len(rows), "n_valid_answers": len(answered), "n_judged": len(judged),
        "answer_accuracy": proportion(len(correct), len(rows)),
        "accuracy_valid_answers_only": proportion(len(correct), len(answered)),
        "reasoning_correctness": mean_score(judged, "reasoning_correctness"),
        "reasoning_completeness": mean_score(judged, "reasoning_completeness"),
        "hallucination_rate": proportion(sum(r["judgment"]["hallucination"] for r in judged), len(judged)),
        "answer_reasoning_consistency": proportion(sum(r["judgment"]["consistency"] for r in judged), len(judged)),
        "false_reasoning_rate": proportion(sum(r["judgment"]["material_error"] for r in correct_judged), len(correct_judged)),
        "correct_answer_material_error_joint_rate": proportion(sum(r["judgment"]["material_error"] for r in correct_judged), len(judged)),
        "judgment_coverage": proportion(len(judged), len(answered)),
        "confusion_matrix": confusion,
        "balanced_accuracy_observed_classes_percent": round(100*sum(recalls)/len(recalls), 2) if recalls else None,
        "classes_present": len(recalls),
        "label_counts": dict(Counter(r["gold_answer"] for r in rows)),
        "majority_class_baseline_percent": round(100*max(Counter(r["gold_answer"] for r in rows).values())/len(rows), 2) if rows else None,
    }
    if not any(r.get("answer_status") in {"ok", "error"} for r in rows):
        result["answer_accuracy"] = {"value_percent": None, "numerator": 0,
                                     "denominator": len(rows), "ci95_percent": None}
    return result


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def display(value):
    if "mean" in value:
        return "N/A" if value["mean"] is None else "%.2f / 5 (n=%d)" % (value["mean"], value["n"])
    if value["value_percent"] is None:
        return "N/A (no assessed responses)"
    low, high = value["ci95_percent"]
    return "%.1f%% (%d/%d); 95%% CI %.1f–%.1f%%" % (value["value_percent"], value["numerator"], value["denominator"], low, high)


def report(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    sample = json.loads((run_dir / "sample.json").read_text())
    rows = []
    for item in sample:
        result_file = run_dir / "records" / (item["id"] + ".json")
        result = json.loads(result_file.read_text()) if result_file.exists() else {}
        rows.append(dict(item, **result))
    computed = metrics(rows)
    mode = manifest["mode"]
    status = "COMPLETE" if computed["n_valid_answers"] == len(rows) and (
        not manifest["config"]["judge_enabled"] or computed["n_judged"] == len(rows)) else "PARTIAL / NOT RUN"
    warning = ("SYNTHETIC SMOKE TEST — these are fixtures, not model results." if mode == "smoke" else
               "Automated pilot only. Explanation scores use the same model in a separate judging call; not clinician-validated.")
    usage = Counter()
    for path in (run_dir / "raw").glob("*.response.json"):
        raw = json.loads(path.read_text())
        for key in ["input_tokens", "output_tokens", "total_tokens"]:
            local_usage = {"input_tokens": raw.get("prompt_eval_count", 0),
                           "output_tokens": raw.get("eval_count", 0),
                           "total_tokens": raw.get("prompt_eval_count", 0) + raw.get("eval_count", 0)}
            usage[key] += raw.get("usage", local_usage).get(key, 0)
    output = {"status": status, "mode": mode, "warning": warning,
              "model_requested": manifest["config"]["model"],
              "models_returned": sorted({r["model_returned"] for r in rows if r.get("model_returned")}),
              "usage_recorded_all_attempts": dict(usage), "metrics": computed}
    save_json(run_dir / "metrics.json", output)
    flat = []
    for r in rows:
        candidate, judge = r.get("prediction", {}), r.get("judgment", {})
        flat.append({"id": r["id"], "gold_answer": r["gold_answer"], "answer": candidate.get("answer", ""),
                     "accuracy": int(candidate.get("answer") == r["gold_answer"]),
                     "explanation": candidate.get("explanation", ""),
                     "answer_status": r.get("answer_status", "pending"),
                     "judge_status": r.get("judge_status", "pending"),
                     **{k: judge.get(k, "") for k in ["reasoning_correctness", "reasoning_completeness", "hallucination", "consistency", "material_error"]},
                     "judge_assessment": judge.get("assessment", ""), "mode": mode})
    if flat:
        write_csv(run_dir / "results.csv", flat, list(flat[0]))
    review = []
    for r in rows:
        if r.get("answer_status") != "ok":
            continue
        review.append({"id": r["id"], "question": r["question"],
                       "abstract": "\n\n".join(r["contexts"]), "reference_answer": r["gold_answer"],
                       "reference_conclusion": r["reference_explanation"],
                       "candidate_answer": r["prediction"]["answer"],
                       "candidate_explanation": r["prediction"]["explanation"],
                       **{k: "" for k in ["reviewer_id", "reasoning_correctness", "reasoning_completeness", "hallucination", "consistency", "material_error", "notes"]}})
    # A template is regenerated; completed annotations belong in a DIFFERENT file.
    if review:
        write_csv(run_dir / "human_review_TEMPLATE.csv", review, list(review[0]))
    labels = [("Answer accuracy ↑", "answer_accuracy"),
              ("Reasoning correctness ↑ (automated)", "reasoning_correctness"),
              ("Reasoning completeness ↑ (automated)", "reasoning_completeness"),
              ("Hallucination rate ↓ (automated)", "hallucination_rate"),
              ("Answer–reasoning consistency ↑ (automated)", "answer_reasoning_consistency"),
              ("False reasoning ↓ (correct answers; automated)", "false_reasoning_rate")]
    metric_table = "\n".join("| %s | %s |" % (name, display(computed[key])) for name, key in labels)
    introduction = ("# PubMedQA pilot report\n\n**%s**\n\nStatus: **%s**\n\nModel requested: `%s`. "
                    "Valid answers: %d/%d. Judged explanations: %d.\n\n"
                    "This is a seeded development sample from PQA-L, not the official test split.\n\n" %
                    (warning, status, manifest["config"]["model"], computed["n_valid_answers"], len(rows), computed["n_judged"]))
    md = introduction + "| Metric | Result |\n|---|---|\n" + metric_table + "\n\n"
    md += ("Accuracy counts failed/missing answers as incorrect; conditional valid-only accuracy is in metrics.json. "
           "Explanation metrics exclude unassessed responses and show their denominators. False reasoning means "
           "a material explanation error among correctly answered, successfully judged items. Confidence intervals "
           "are Wilson intervals and do not include judge error. Ordinal score distributions are in metrics.json.\n\n"
           "Completeness is relative to the abstract/reference conclusion, not an expert-authored reasoning rubric. "
           "Hallucination is evidence-relative; this pipeline does not verify external medical knowledge or citations. "
           "Public-data training exposure is unknown. These observations do not establish clinical safety or model superiority.\n\n")
    md += "## Confusion matrix\n\n| Gold / predicted | yes | no | maybe | missing/invalid |\n|---|---|---|---|---|\n"
    for gold, counts in computed["confusion_matrix"].items():
        md += "| %s | %d | %d | %d | %d |\n" % (gold, counts["yes"], counts["no"], counts["maybe"], counts["invalid_or_missing"])
    md += "\n## Files\n\n- results.csv: per-question results\n- metrics.json: metrics, denominators, uncertainty and usage\n- sample.json / manifest.json: frozen data and settings\n- raw/: request and response audit trail\n- human_review_TEMPLATE.csv: blank independent review sheet, available after valid answers\n"
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    e = html.escape
    table = "".join("<tr><th>%s</th><td>%s</td></tr>" % (e(name), e(display(computed[key]))) for name, key in labels)
    cards = []
    for r in rows:
        pred = r.get("prediction", {})
        assessment = r.get("judgment", {})
        cards.append("<details><summary>%s · gold %s · predicted %s</summary><h3>Question</h3><p>%s</p>"
                     "<h3>Abstract</h3><p>%s</p><h3>Explanation</h3><p>%s</p>"
                     "<h3>Reference conclusion</h3><p>%s</p><h3>Automated assessment</h3><pre>%s</pre></details>" %
                     (e(r["id"]), e(r["gold_answer"]), e(pred.get("answer", "pending")), e(r["question"]),
                      e("\n\n".join(r["contexts"])), e(pred.get("explanation", "Not generated")),
                      e(r["reference_explanation"]), e(json.dumps(assessment, indent=2, ensure_ascii=False))))
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PubMedQA pilot results</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 24px;color:#172d3b;background:#f5f8fa}h1{font-size:32px}.notice{padding:16px;background:#fff0c9;border-left:5px solid #bc7900}table{border-collapse:collapse;width:100%%;background:white}th,td{text-align:left;padding:12px;border-bottom:1px solid #dbe3e8}details{padding:16px;margin:12px 0;background:white;border:1px solid #dbe3e8;border-radius:8px}summary{cursor:pointer;font-weight:600}p,pre{white-space:pre-wrap;overflow-wrap:anywhere}pre{font-size:13px}a{color:#165780}</style>
<h1>PubMedQA pilot results</h1><p class="notice">%s</p><p>Status: <strong>%s</strong> · Model: %s · Answered: %d/%d · Judged: %d</p>
<table>%s</table><p>Seeded pilot sample, not the official test split. Accuracy includes missing answers as failures. Explanation metrics use assessed responses only. Intervals are Wilson 95%% CIs; they exclude judge uncertainty. False reasoning is conditional on correct answers. See <a href="report.md">method notes</a> and <a href="metrics.json">all metrics</a>.</p>
<h2>Inspect every question</h2>%s</html>""" % (e(warning), e(status), e(manifest["config"]["model"]), computed["n_valid_answers"], len(rows), computed["n_judged"], table, "".join(cards))
    (run_dir / "report.html").write_text(page, encoding="utf-8")
    return output
