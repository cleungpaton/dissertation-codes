# dissertation-codes

Code and data for my MATH3001 dissertation: **"Does Distillation Preserve Mathematical Reasoning? Evaluating Reasoning Fidelity in Compressed Language Models"** (University of Leeds, 2025/26).

The project compares a 32B teacher model against a 1.5B distilled student from the DeepSeek-R1 family, testing whether knowledge distillation preserves mathematical reasoning behaviour or just preserves final-answer accuracy. 40 maths problems across four difficulty levels, 5 runs each, 600 total outputs. All inference was done locally through Ollama on a MacBook Pro M3 Max.

Two main findings:
- Verification behaviour (the model checking its own work) degrades before accuracy does. At Levels 1-2 the student matches the teacher's accuracy but barely verifies.
- When both models get a problem wrong, they tend to make the same type of error 42% of the time, compared to 8% for an independently trained baseline. Distillation transfers failure modes, not just capability.

## Scripts

| Script | What it does | README |
|--------|-------------|--------|
| `run_experiment.py` | Sends problems to models via Ollama, records full outputs, extracts and scores answers | [run_experiment_README.md](run_experiment_README.md) |
| `rescore.py` | Re-extracts and re-scores existing outputs with improved matching logic. No model calls needed. | [rescore_README.md](rescore_README.md) |
| `extract_features.py` | Extracts the V1-V5 reasoning features (verification, hesitation, contradiction, symbolic engagement, structure) from each output | [extract_features_README.md](extract_features_README.md) |
| `h3_regression.py` | Applies manual review corrections then runs the logistic regression testing whether the accuracy gap widens with difficulty | [h3_regression_README.md](h3_regression_README.md) |
| `compute_fidelity.py` | Computes the composite fidelity score F(x). This metric failed due to a length effect and was abandoned, but the code is included for transparency. | [compute_fidelity_README.md](compute_fidelity_README.md) |

## Data

| File | Description |
|------|-------------|
| `questions_clean.csv` | The 40 maths problems used in the experiment, with problem IDs, question text, gold answers, complexity levels, domains, and source datasets (GSM8K, Hendrycks MATH, OlympiadBench) |

## Pipeline order

```
1. run_experiment.py      Generate model outputs and initial scores
2. rescore.py             Fix answer extraction errors
3. Manual review          Read remaining flagged outputs, correct by hand in Excel
4. extract_features.py    Extract V1-V5 reasoning features
5. h3_regression.py       Logistic regression on corrected accuracy
6. compute_fidelity.py    Composite fidelity score (abandoned)
```

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com) installed and running locally (only needed for step 1)
- Python packages: `pandas`, `requests`, `openpyxl`, `sympy`, `numpy`, `statsmodels`

```
pip install pandas requests openpyxl sympy numpy statsmodels
```

## Models

| Role | Model | Parameters |
|------|-------|------------|
| Teacher | DeepSeek-R1-Distill-Qwen-32B (Q4_K_M) | 32B |
| Student | DeepSeek-R1-Distill-Qwen-1.5B | 1.5B |
| Baseline | Qwen2.5-1.5B | 1.5B |

Pull them via Ollama before running the experiment:

```
ollama pull hengwen/DeepSeek-R1-Distill-Qwen-32B:q4_k_m
ollama pull erwan2/DeepSeek-R1-Distill-Qwen-1.5B
ollama pull qwen2.5:1.5b
```
