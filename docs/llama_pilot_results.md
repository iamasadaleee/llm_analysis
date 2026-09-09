# Completed free local pilot — September 9, 2026

Model: Llama 3.2 3B (`llama3.2:3b`, 3.2B parameters, Q4_K_M quantization).
Runtime: local Ollama 0.5.12. No API key or paid inference service was used.
Model digest: `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`.

Dataset: the downloaded PubMedQA PQA-L pool of 1,000 questions. This run used the
preselected simple random sample of 10 questions, seed 42, without changing prompts
or the sample based on observed model performance. These are development-pilot results,
not an official test-set benchmark. All 20 inference requests completed: 10 answering
calls and 10 separate same-model judging calls. There were no failed or missing outputs.

| Metric | Observed result |
|---|---|
| Answer accuracy | 60% (6/10); Wilson 95% CI 31.27–83.18% |
| Automated reasoning correctness | 4.5/5 across 10 explanations |
| Automated reasoning completeness | 4.3/5 across 10 explanations |
| Automated hallucination flags | 0/10 |
| Automated answer–explanation consistency | 80% (8/10) |
| Automated material-error flags among correct answers | 0/6 |

The sample contains six gold yes answers and four gold no answers, with no gold maybe
answers. The model correctly answered all six yes questions and missed all four no
questions (predicting maybe three times and yes once). Always predicting yes would
also achieve 60% on this sample. A larger, prespecified sample covering all classes is
needed before interpreting model performance.

The model judged its own explanations in independent calls. These scores are not
clinician-validated; zero flagged hallucinations or material errors does not establish
their absence. Some written assessments mention weaknesses despite high scores,
illustrating why manual review is necessary. The interval for 0/10 hallucination flags
extends to 27.75%; the interval for 0/6 material-error flags extends to 39.03%, even
before considering judge error. High explanation scores do not override incorrect answers.

Recorded usage: 11,971 input tokens and 2,909 generated tokens. No API charges.

## Inspect or reproduce

- `runs/pubmedqa_llama32_3b_pilot/report.html`: expandable question-level report
- `runs/pubmedqa_llama32_3b_pilot/results.csv`: per-question results
- `runs/pubmedqa_llama32_3b_pilot/metrics.json`: metrics and denominators
- `runs/pubmedqa_llama32_3b_pilot/raw/`: all original requests and responses
- `runs/pubmedqa_llama32_3b_pilot/human_review_TEMPLATE.csv`: independent review template
- `configs/llama_pilot.json`: exact pilot configuration

```bash
python3 -m medpilot run --config configs/llama_pilot.json
```

Completed calls are reused. To create a fresh experiment, copy the configuration and
change output_dir; change sample_size if expanding the pilot. Do not mix this pilot
with the earlier synthetic smoke-test results.

Software validation: 15 tests passed, including local API parsing, truncation rejection,
gold-answer exclusion, metric denominators and failed-call recovery. The completed
run was also checked for 20 raw responses, 10 completed records and generated exports.
