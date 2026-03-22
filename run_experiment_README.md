# run_experiment.py

Main experiment runner for my MATH3001 dissertation project: "Does Distillation Preserve Mathematical Reasoning?"

This script sends maths problems to language models running locally through Ollama, records their full chain-of-thought outputs, extracts final answers, and scores them against known solutions. It was used to generate all 600 model outputs (3 models x 40 problems x 5 runs) analysed in the dissertation.

## What it does

1. Reads a CSV of maths problems (each with a problem ID, question text, and known answer)
2. Sends each problem to one or more locally hosted LLMs via the Ollama API
3. Records the full model output, including the entire reasoning trace
4. Extracts the final answer using regex (looks for `FINAL ANSWER:`, `\boxed{...}`, or falls back to the last line)
5. Scores the extracted answer against the gold answer using three levels of matching: exact string, numeric comparison, and symbolic comparison via SymPy
6. Writes every API call to a JSONL log file as it goes, so nothing is lost if the process crashes partway through
7. Produces summary CSVs with accuracy breakdowns by model and complexity level
8. Flags answers that could not be matched automatically for manual review

## Models

The script is configured for three models from the DeepSeek-R1 distillation family:

- **Teacher**: `hengwen/DeepSeek-R1-Distill-Qwen-32B:q4_k_m` (32B parameters, quantised)
- **Student**: `erwan2/DeepSeek-R1-Distill-Qwen-1.5B` (1.5B parameters)
- **Baseline**: `qwen2.5:1.5b` (1.5B parameters, independently trained, not distilled)

These are set in the `MODELS` dictionary near the top of the file. You can change them to any model available in your Ollama installation.

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com) installed and running locally
- The models you want to use pulled via `ollama pull <model_name>`

Python packages:

```
pip install pandas requests openpyxl sympy
```

On macOS with system Python you may need `--break-system-packages`.

## Input format

The script expects a CSV (or Excel file) with at least these columns:

| Column | Description |
|--------|-------------|
| `problem_id` | Unique identifier for each problem (e.g. `L1_GSM8K_01`) |
| `question` | The full problem text |
| `final_answer` | The known correct answer |

Optional columns that will be carried through to the output if present: `dataset`, `complexity_level`, `domain`, `structural_type`, `source_split`.

## Usage

Run the full experiment (all models, 5 runs each):

```bash
python3 run_experiment.py --input questions_clean.csv --output results/ --runs 5
```

Run a single model:

```bash
python3 run_experiment.py --input questions_clean.csv --output results_student/ --runs 5 --models student
```

Quick test with 2 problems:

```bash
python3 run_experiment.py --input questions_clean.csv --output test_results/ --runs 1 --limit 2
```

Run specific models:

```bash
python3 run_experiment.py --input questions_clean.csv --output results/ --runs 5 --models teacher,student
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | (required) | Path to the CSV or Excel file containing problems |
| `--output` | `results/` | Directory where output files will be saved |
| `--runs` | 5 | Number of times to run each problem per model |
| `--models` | `all` | Which models to run: `all`, `teacher`, `student`, `baseline`, or comma-separated |
| `--limit` | None | Only run the first N problems (useful for testing) |

## Output files

The script creates several files in the output directory:

| File | Contents |
|------|----------|
| `run_log.jsonl` | One JSON object per API call. Contains the full model output, extracted answer, gold answer, correctness score, timing, and all metadata. This is the primary data file. |
| `summary.csv` | Same information as the JSONL but in CSV format, without the full output text (too large for CSV). |
| `accuracy_by_complexity.csv` | Accuracy grouped by model and complexity level. |
| `accuracy_overall.csv` | Overall accuracy per model. |
| `needs_manual_review.csv` | Answers where the automatic matcher could not determine correctness. These need a human to read the full output and judge. |
| `experiment_config.json` | Full configuration including model names, decoding parameters, prompt template, and timestamp. |

## Decoding parameters

These are set as constants near the top of the file:

| Parameter | Value | Reason |
|-----------|-------|--------|
| Temperature | 0.6 | High enough that repeated runs produce different outputs, low enough for coherent reasoning |
| Top-p | 0.95 | Standard nucleus sampling |
| Max tokens | 4096 | Practical limit imposed by memory on the hardware used |

## Answer matching

The script tries three methods to check if the model's answer matches the gold answer:

1. **Exact string match** after normalising both answers (stripping LaTeX formatting, whitespace, dollar signs, etc.)
2. **Numeric match** by parsing both as numbers and comparing with a tolerance of 1e-6
3. **Symbolic match** using SymPy to check mathematical equivalence (e.g. `2/9` vs `0.222...`)

If none of these methods produce a match, the answer is flagged for manual review. In practice, about 15-20% of answers needed manual review due to formatting differences the normaliser could not handle.

## Notes

- The JSONL log file is written incrementally (one line per API call, flushed immediately). If the script crashes or you interrupt it, all completed calls are saved.
- The teacher model (32B) takes 4-8 minutes per call on an M3 Max MacBook Pro. A full run of 200 calls takes roughly 12-15 hours. Consider running overnight with `caffeinate -i` on macOS to prevent sleep.
- The script appends to the JSONL file rather than overwriting it. If you re-run, you will get duplicate entries. Delete or rename the existing log file before re-running, or use a different output directory.
