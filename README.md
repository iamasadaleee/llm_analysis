# Medical LLM study: complete pilot pipeline

This project follows your proposed study: answer accuracy, reasoning correctness,
reasoning completeness, hallucination rate, answer–reasoning consistency, and false reasoning.
The first experiment uses **PubMedQA PQA-L + GPT-5**. No training or fine-tuning is needed.

The real 1,000-question dataset is downloaded. **A free local Llama 3.2 3B pilot is
available through Ollama**, alongside the original optional GPT-5 configuration.
Synthetic smoke tests are software checks, never study results.

## Free local model: no API key

Ollama is installed on this machine. Start its server in a terminal (leave it running),
then run the remaining commands in another terminal:

```bash
ollama serve
```

```bash
ollama pull llama3.2:3b
python3 -m medpilot run --config configs/llama_pilot.json
open runs/pubmedqa_llama32_3b_pilot/report.html
```

The download is approximately 2 GB and is cached; inference stays on your computer.
There are no API fees, though local compute uses memory and electricity. The existing
Llama 3.3 70B model is too large for a practical pilot on this 18 GB Mac, so this pilot
uses Llama 3.2 3B. This small general-purpose model is a pipeline baseline, not a
medical specialist or a replacement for testing stronger models in the final paper.

The same model answers 10 questions and evaluates the explanations in 10 separate
requests. Model identity/digest, Ollama version, quantization details, exact prompts,
raw outputs, token counts and timing are recorded. Local settings: temperature 0,
seed 42, 8,192-token context, and at most 1,024 generated tokens per request.
`reasoning_effort` is an OpenAI-only setting and is not sent to Ollama. Local requests
are not automatically retried. Resume a failed call explicitly with `--retry-failed`.
Same-model judging is exploratory and can miss mistakes or overrate explanations.

Local API reference: https://docs.ollama.com/api/chat

## Optional GPT-5 quick start

Requires Python 3.9+; **no third-party packages are required**. Run from this folder.

```bash
cp .env.example .env
# Edit .env locally and set OPENAI_API_KEY. Do not paste the key into chat.
python3 -m medpilot doctor
python3 -m medpilot run
```

`run` downloads/verifies data, samples 10 questions, freezes the protocol, requests an
answer and explanation, makes a separate judging call, scores, and exports reports.
It makes up to **20 logical model calls** by default. HTTP retries can add calls.
Each call allows 4,096 output tokens, including reasoning tokens; this is a token limit,
not a dollar budget. Requests can incur API charges. This pilot does not automatically
switch models if account access fails. Set a spending limit in your provider account.

Open the result in your browser:

```bash
open runs/pubmedqa_gpt5_pilot/report.html
```

The answer and judge both use GPT-5, in independent requests without conversation history.
This keeps the pilot to one AI model, but introduces **same-model judge bias**. Explanation
metrics are automated, evidence-relative estimates, not validated clinical reasoning scores.

## Run stages separately

```bash
python3 -m medpilot download
python3 -m medpilot prepare
python3 -m medpilot run
python3 -m medpilot report
```

`prepare` produces an unrun report with pending responses. It does not call the model.
`doctor` reports whether a key exists without displaying it; it does not check account access.
The `.env` parser handles NAME=value and quoted values, not shell scripts or interpolation.

An offline software check, after downloading the data:

```bash
python3 -m unittest discover -s tests -v
python3 -m medpilot smoke
```

Smoke reports go in `runs/pubmedqa_gpt5_pilot_SYNTHETIC_SMOKE/` and carry a prominent
synthetic label. The tests use fake HTTP responses and do not incur charges.

## Experiment configuration

Edit `configs/pilot.json`. For a different experiment, also choose a **new output_dir**.

| Field | Purpose |
|---|---|
| sample_size | Start at 10; expand to 50 after reviewing pilot outputs |
| seed | Reproducible simple random sample, default 42 |
| provider | `openai` for the API or `ollama` for local inference |
| model | OpenAI model ID or installed Ollama tag |
| reasoning_effort | GPT-5 setting, default low |
| max_output_tokens | Budget per call; incomplete outputs are not scored |
| judge_enabled | Enable your explanation metrics; false gives answer accuracy only |
| max_attempts | Bounded retries for transient HTTP failures |
| output_dir | Separate directory for every distinct experiment |

The requested ID and model returned by the API are saved. An alias may change over time;
the returned ID is recorded but does not itself guarantee reproducibility. No temperature
or seed is sent to GPT-5; the sample seed is **not** an inference determinism guarantee.
Model availability depends on your account. If `gpt-5` is unavailable, explicitly edit
the model setting and use a new output directory; no automatic substitution happens.

## Dataset and leakage controls

- Official source: https://github.com/pubmedqa/pubmedqa
- Frozen revision: `1cbae8e92f72f20c8d3747cbb3bf5bc53554d997`
- File: `data/ori_pqal.json`, 1,000 labeled questions (552 yes / 338 no / 110 maybe).
- SHA-256: `8b3276be8942ebbd77f3ddcda12c1749bf0e490045a736fd8438ee40cf37a41d`
- Local license: `data/raw/LICENSE.pubmedqa`; source/provenance: `data/raw/provenance.json`.
- Paper: https://aclanthology.org/D19-1259/

This is biomedical research QA (including some nonclinical research), not exclusively
patient diagnosis. Input is the question and abstract sections without the conclusion.
The solver prompt is constructed using an explicit field allowlist: gold labels,
LONG_ANSWER, existing prediction fields, and identifiers are not passed to the solver.
The separate evaluator receives the gold label and reference conclusion after generation.
Schema validation rejects missing sections, conclusion-labeled sections, duplicate
question/context pairs, and invalid labels. It cannot prove absence of training exposure
or semantic answer clues in legitimate abstract results.

The pilot is a simple random sample from the entire PQA-L pool, **not the official test
split**. It is for developing the pipeline. Freeze an appropriate held-out split and keep
pilot questions separate before reporting a final paper benchmark. The label distribution
of a small sample can differ markedly from the whole dataset.

## Metrics and denominators

| Metric | Definition |
|---|---|
| Answer accuracy | Gold-label matches / all selected questions; failed or missing answers count as unsuccessful |
| Reasoning correctness | Mean anchored 1–5 score among successfully judged explanations |
| Reasoning completeness | Mean anchored 1–5 coverage score relative to supplied evidence/reference |
| Hallucination rate | Judged explanations with an unsupported/invented assertion / judged explanations |
| Answer–reasoning consistency | Explanations that support their own answer / judged explanations |
| False reasoning | Correctly answered, judged items with a material explanation error / correctly answered, judged items |

Every proportion includes numerator, denominator, and a Wilson 95% confidence interval.
Zero denominators are N/A, never zero. Intervals do not account for judge unreliability.
Correctness/completeness include ordinal score distributions. The full report also gives
valid-answer-only accuracy, judgment coverage, the joint correct-answer/error rate,
label counts, majority-class baseline, and a confusion matrix including missing answers.
Balanced accuracy is averaged over observed classes only; classes_present is reported.
Partial/unrun reports must not be interpreted as completed model performance.

The exact scoring rubric is in `medpilot/prompts.py` and copied into each run.
Consistency is distinct from accuracy: an incorrect answer can have an internally
consistent explanation. Hallucination and material error can overlap. Completeness
does not require a differential diagnosis for a research yes/no/maybe question.

## Outputs

```text
data/raw/                       downloaded dataset, license, provenance
data/processed/                 normalized data, sample, data audit
runs/pubmedqa_gpt5_pilot/
  manifest.json                frozen settings, data/code hashes, sample IDs, timestamps
  prompts.json                 exact prompts and schemas
  sample.json                  frozen questions and references
  records/                     resumable per-question answers, grades, status, usage
  raw/                         exact request bodies and every received response/HTTP error
  results.csv                  one row per question
  metrics.json                 results, denominators, CIs, all recorded token usage
  report.md                    readable methods/results
  report.html                  standalone report with expandable question details
  human_review_TEMPLATE.csv    blank review form, created after valid answers
```

Raw request files do not contain API credentials. API responses and requests contain
public biomedical study text; they are saved locally and ignored by Git by default.
Token totals include received responses from failed parsing/incomplete-output attempts.
Timeouts may have been billed without returning usage; reported usage is not a billing statement.

## Resume and failure handling

Successful calls are cached. Run the same command again to skip them. If a call fails,
the run stops promptly, preserves completed work and produces a partial report.
After fixing the cause, explicitly retry failed calls:

```bash
python3 -m medpilot run --retry-failed
```

Auth/model-access errors are not retried automatically. Rate limits and transient HTTP
server errors get bounded retries. Network timeouts are not automatically retried because
completion/billing may be unknown. Refusals, invalid JSON/schema, and truncated outputs
are failures, not wrong labels inferred from surrounding text. Increase output tokens
in a **new run** if responses repeatedly exhaust the budget.

Runs reject changed configuration, sample, prompt, or code hashes to prevent mixing
experiments. A process lock prevents concurrent writing. After a hard crash, check that
the process has stopped before deleting `.run.lock`. Completed human annotations should
be saved to a new filename; report regeneration rewrites only the blank TEMPLATE file.

## Before using results in a publication

Inspect the question-level outputs, especially yes/no/maybe ambiguity and reference issues.
Have two medical reviewers independently fill separate copies of the review template,
without seeing automated scores; record agreement and adjudicate disagreements. This
pilot exports annotations but does not automatically turn unvalidated human files into
adjudicated ground truth. A reference conclusion is not a full expert reasoning rubric.

Use pilot results to finalize prompts and metrics, then run a held-out evaluation with
an adequately justified sample size and additional models from your proposed study.
Do not claim hidden-reasoning faithfulness, clinical readiness, or model superiority from
this one-model pilot. No result values are prefilled from the example table in your proposal.

API implementation references:
- https://developers.openai.com/api/docs/guides/structured-outputs
- https://developers.openai.com/api/docs/models/gpt-5
