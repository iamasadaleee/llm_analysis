import hashlib
import json
import random
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REVISION = "1cbae8e92f72f20c8d3747cbb3bf5bc53554d997"
CHECKSUM = "8b3276be8942ebbd77f3ddcda12c1749bf0e490045a736fd8438ee40cf37a41d"
BASE_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/" + REVISION


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def download(root):
    folder = Path(root) / "data/raw"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "ori_pqal.json"
    if not target.exists():
        request = urllib.request.Request(BASE_URL + "/data/ori_pqal.json", headers={"User-Agent": "medpilot/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
        if digest(raw) != CHECKSUM:
            raise ValueError("Downloaded dataset checksum mismatch")
        target.write_bytes(raw)
    if digest(target.read_bytes()) != CHECKSUM:
        raise ValueError("Cached dataset checksum mismatch; preserve it elsewhere and download again")
    license_file = folder / "LICENSE.pubmedqa"
    if not license_file.exists():
        with urllib.request.urlopen(BASE_URL + "/LICENSE", timeout=60) as response:
            license_file.write_bytes(response.read())
    save_json(folder / "provenance.json", {
        "dataset": "PubMedQA PQA-L", "revision": REVISION,
        "url": BASE_URL + "/data/ori_pqal.json", "sha256": CHECKSUM,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "LICENSE.pubmedqa", "rows": 1000,
        "paper": "https://aclanthology.org/D19-1259/",
    })
    return target


def normalize(raw):
    rows, seen = [], set()
    for identifier, item in sorted(raw.items()):
        if item["final_decision"] not in {"yes", "no", "maybe"}:
            raise ValueError("Unknown label for " + identifier)
        contexts = item["CONTEXTS"]
        labels = item["LABELS"]
        if not contexts or len(contexts) != len(labels):
            raise ValueError("Missing or mismatched abstract sections: " + identifier)
        if any("CONCLU" in label.upper() for label in labels):
            raise ValueError("Conclusion section in solver input: " + identifier)
        if not all(isinstance(c, str) and c.strip() for c in contexts):
            raise ValueError("Empty context: " + identifier)
        if not item["QUESTION"].strip() or not item["LONG_ANSWER"].strip():
            raise ValueError("Missing question or reference: " + identifier)
        fingerprint = digest(json.dumps([item["QUESTION"], contexts], sort_keys=True).encode())
        if fingerprint in seen:
            raise ValueError("Duplicate question/context: " + identifier)
        seen.add(fingerprint)
        rows.append({"id": identifier, "question": item["QUESTION"], "contexts": contexts,
                     "gold_answer": item["final_decision"],
                     "reference_explanation": item["LONG_ANSWER"]})
    return rows


def prepare(root, size, seed):
    raw_path = download(root)
    rows = normalize(json.loads(raw_path.read_text(encoding="utf-8")))
    if not 1 <= size <= len(rows):
        raise ValueError("sample_size must be between 1 and %d" % len(rows))
    # Simple random sample, independent of gold labels. Pilot only, not official test split.
    sample = random.Random(seed).sample(rows, size)
    folder = Path(root) / "data/processed"
    save_json(folder / "all.json", rows)
    save_json(folder / ("sample_%s_seed_%s.json" % (size, seed)), sample)
    save_json(folder / "audit.json", {
        "total_rows": len(rows), "label_counts": dict(Counter(r["gold_answer"] for r in rows)),
        "sample_rows": size, "sample_label_counts": dict(Counter(r["gold_answer"] for r in sample)),
        "seed": seed, "sampling": "simple random sample from PQA-L; NOT official test split",
        "duplicate_question_contexts": 0, "conclusion_sections_in_input": 0,
    })
    return sample
