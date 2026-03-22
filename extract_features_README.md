# extract_features.py

Extracts the V1-V5 reasoning feature vector from each model output. This is the script that produces the data behind the H1 analysis. It reads the JSONL log from `run_experiment.py` (or the rescored version from `rescore.py`) and scores each output on five dimensions of reasoning behaviour.

## The five features

| Feature | Name | What it detects | Encoding |
|---------|------|----------------|----------|
| V1 | Verification | Did the model check its own work? Looks for substitution back into equations, domain checking, extraneous root testing, explicit "let me verify" language. | Binary (0/1) |
| V2 | Hesitation | Signs of reasoning instability: "wait", "actually", "let me try a different approach", "I made a mistake", "hmm". | Binary (0/1) |
| V3 | Contradiction | Logical inconsistencies within the same output. The model says one thing and then contradicts it later. | Binary (0/1) |
| V4 | Symbolic engagement | How much mathematical notation the model uses. Counts lines containing equations, LaTeX fractions, roots, integrals, exponents, etc. Capped at 4, then divided by 4. | Continuous (0-1) |
| V5 | Solution structure | How organised the output is. Checks three things: are there numbered steps or sequencing words, is there a clearly marked final answer, and is the output a reasonable length (not a one-liner, not absurdly long). Score is the average of those three binary checks. | Continuous (0, 0.33, 0.67, or 1.0) |

V1 is the primary feature for the H1 hypothesis test. V2-V5 provide supporting context but are not used in any formal statistical test.

## How detection works

Each feature uses regex pattern matching against the full model output text. V1, for example, matches against 14 patterns covering explicit verification language ("let me verify", "substituting back"), domain checking ("check the domain"), extraneous root handling ("discard this root"), and general sense-checking ("does this make sense", "sanity check").

The automated V1 scoring was validated against manual human scoring on all 40 Run 1 outputs, producing a Cohen's Kappa of 1.0 (perfect agreement).

## Requirements

- Python 3.8+
- `pandas`

```
pip install pandas
```

No Ollama or network access needed. This script only reads local files.

## Usage

```bash
python3 extract_features.py --input results_teacher_rescored/run_log_rescored.jsonl --output teacher_features.csv
```

Or with the student:

```bash
python3 extract_features.py --input results_student_rescored/run_log_rescored.jsonl --output student_features.csv
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | (required) | Path to a `run_log.jsonl` or `run_log_rescored.jsonl` file |
| `--output` | `results/features.csv` | Path for the output CSV |

## Output

A CSV file with one row per (problem_id, model_name, run_id). Columns include:

- `problem_id`, `model_name`, `run_id`, `correct`, `complexity_level`, `domain`, `dataset` (metadata carried through from the log)
- `V1` through `V5` (the feature scores)
- `V1_match_count`, `V1_patterns` (how many V1 patterns matched, and which ones, for debugging)
- Same for V2 and V3
- `V4_raw_count`, `V4_capped` (raw equation count before and after the cap at 4)
- `V5_has_steps`, `V5_has_conclusion`, `V5_reasonable_length`, `V5_char_count` (the three sub-components and the raw character count)

The script also prints summary tables to the terminal when it finishes: feature means by model, and V1 broken down by model and complexity level.

## Example terminal output

```
Reading: results_teacher_rescored/run_log_rescored.jsonl
Saved: teacher_features.csv (200 rows)

--- Feature Means by Model ---
              V1     V2     V3     V4     V5
model_name
teacher    0.700  0.680  0.040  0.954  0.863

--- V1 (Verification) by Model and Complexity ---
model_name  complexity_level
teacher     1                   0.644
            2                   0.767
            3                   0.711
            4                   0.660
```

## Where this fits in the pipeline

1. `run_experiment.py` generates model outputs
2. `rescore.py` corrects answer extraction
3. **`extract_features.py` extracts the V1-V5 reasoning features** (you are here)
4. The output CSV is then used in the H1 analysis (Wilson confidence intervals on V1 rates, the V2-V5 summary table, etc.)

## Known limitation

V1 and V2 are detected by keyword presence, not by whether the model actually followed through on what it said. An output that says "let me verify this" and then goes off on an unrelated tangent scores V1 = 1, same as an output that genuinely substitutes the answer back and confirms it. This length-related inflation is discussed in the dissertation as the reason the composite fidelity score F(x) was abandoned. At Levels 1-2, where the student barely verifies at all, this distinction does not affect the results. At Levels 3-4, where both models produce long outputs, it makes V1 less reliable.
