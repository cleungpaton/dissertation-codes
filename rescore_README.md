# rescore.py

Stage 2 of the scoring pipeline. This script re-processes existing model outputs with improved answer extraction and matching logic, without re-running any models. It reads the JSONL log produced by `run_experiment.py`, re-extracts the final answer from each output, re-scores it against the gold answer, and writes corrected output files.

## Why this exists

The initial answer extraction in `run_experiment.py` missed a significant number of correct answers due to formatting issues. Common problems included LaTeX mismatches (`\dfrac{2}{9}` vs `2/9`), nested braces inside `\boxed{}`, dollar signs being treated as currency rather than LaTeX delimiters, and answers buried in sentence-form responses like "Alexis paid $41 for the shoes."

Rather than re-running the full experiment (which takes 12+ hours per model), this script re-processes the saved outputs in seconds. It applies the same three-level matching strategy as `run_experiment.py` but with improved normalisation, plus an additional no-spaces comparison step that catches things like `x^2 + 7x + 10` vs `x^2+7x+10`.

In practice, rescoring changed 15-20% of the teacher's scores from incorrect to correct, mostly due to LaTeX formatting that the original extractor could not handle.

## What it does

1. Reads every entry from an existing `run_log.jsonl` file
2. Re-extracts the final answer from the stored `full_output` field using improved regex logic
3. Re-scores each extracted answer against the gold answer using four matching levels: exact string, exact string without spaces, numeric comparison, and symbolic comparison via SymPy
4. Prints every answer whose score changed (with the old and new extracted answers, so you can verify the change makes sense)
5. Writes a corrected JSONL log, summary CSV, accuracy tables, and a manual review file for any remaining unmatched answers

## Requirements

- Python 3.8+
- `pandas`
- `sympy` (optional, used for symbolic matching)

```
pip install pandas sympy
```

No Ollama or network access needed. This script only reads and re-processes local files.

## Usage

```bash
python3 rescore.py --input results_student/run_log.jsonl --output results_student_rescored/
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--input` | Path to the `run_log.jsonl` file produced by `run_experiment.py` |
| `--output` | Directory where the corrected files will be saved |

## Output files

| File | Contents |
|------|----------|
| `run_log_rescored.jsonl` | Corrected version of the input JSONL with updated `extracted_answer`, `correct`, and `match_method` fields |
| `summary_rescored.csv` | CSV version of the corrected data (without full output text) |
| `accuracy_by_complexity.csv` | Corrected accuracy grouped by model and complexity level |
| `accuracy_overall.csv` | Corrected overall accuracy per model |
| `needs_manual_review.csv` | Answers that still could not be matched after rescoring. These go to Stage 3 (manual review in Excel). |

## Answer matching levels

The script tries four methods in order:

1. **Exact string** after normalisation (strips LaTeX, whitespace, dollar signs, etc.)
2. **Exact string without spaces** (catches spacing differences in expressions)
3. **Numeric** by parsing both sides as numbers and comparing with tolerance 1e-6
4. **Symbolic** via SymPy, checking mathematical equivalence

If none of these match, the answer is flagged for manual review.

## Example output

When you run it, you will see something like:

```
Reading: results_teacher/run_log.jsonl
Total entries: 200
  ✗→✓ L2_HMATH_10 run 3: '\dfrac{2}{9}' → '2/9' (gold: '2/9') [exact_string]
  ✗→✓ L3_HMATH_07 run 1: '\boxed{\dfrac{21}{8}}' → '21/8' (gold: '21/8') [numeric]

Total scoring changes: 7
```

Each line shows a score that changed, whether it flipped from wrong to right or right to wrong, what the old and new extracted answers were, and which matching method succeeded.

## Where this fits in the pipeline

This is Stage 2 of a three-stage scoring process:

1. **Stage 1** (`run_experiment.py`): Initial automated extraction and scoring during the experiment
2. **Stage 2** (`rescore.py`): Re-extraction with improved logic, no model calls needed
3. **Stage 3** (manual): Remaining unmatched answers reviewed by hand in Excel spreadsheets

The final accuracy numbers reported in the dissertation use the Stage 3 (manually corrected) scores, but Stage 2 caught the majority of extraction errors automatically.
