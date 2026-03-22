# compute_fidelity.py

Computes the composite reasoning fidelity score F(x) that was part of the original research plan. This metric was ultimately abandoned because it produced misleading results, but the script is included here for transparency since the failure is discussed in the dissertation.

## What it does

1. Loads the teacher and student feature CSVs (from `extract_features.py`)
2. Averages each feature (V1-V5) across runs for each problem, then across problems at each complexity level, producing a single 5-element vector per model per level
3. Computes the mean absolute difference between the teacher and student vectors at each level
4. Calculates F(x) = 1 - distance, where a score of 1.0 means the student's reasoning is identical to the teacher's and 0.0 means maximum divergence
5. Prints the full vectors, differences, and F(x) at each level, plus an overall score
6. Saves everything to `fidelity_results.csv`

## Why it failed

The expectation was that F(x) would decrease with difficulty, showing that the student's reasoning diverges further from the teacher's on harder problems. Instead, F(x) went up: 0.725 at Level 1, 0.899 at Level 3, 0.857 at Level 4.

The problem is a length effect in V1 and V2. On harder problems, the student produces much longer outputs as it struggles through extended reasoning chains. Those longer outputs contain more verification and hesitation keywords by sheer volume of text, even when the model is not genuinely verifying or deliberating. This inflates the student's V1 and V2 toward the teacher's values at higher levels, shrinking the apparent distance between the two vectors. The remaining features (V4, V5) are near ceiling for both models at all levels and contribute almost no discriminative signal. V3 is sparse and noisy.

The result is a composite metric built from two features contaminated by a length effect and three features with almost no variance. It cannot distinguish genuine reasoning similarity from superficial keyword overlap.

The complexity interaction was instead tested directly via logistic regression on corrected accuracy data (see `h3_regression_corrected.py`), which avoids the length effect entirely.

## Requirements

- Python 3.8+
- `pandas`, `numpy`

```
pip install pandas numpy
```

## Usage

```bash
python3 compute_fidelity.py --teacher teacher_features.csv --student student_features.csv
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--teacher` | Path to the teacher's feature CSV (from `extract_features.py`) |
| `--student` | Path to the student's feature CSV (from `extract_features.py`) |

## Output

Prints the teacher and student feature vectors at each complexity level, the element-wise absolute differences, the distance, and F(x). Also saves `fidelity_results.csv` with columns for each feature value, the distance, and the fidelity score at each level.

## Example terminal output

```
--- Level 1 ---
  Teacher vector: [0.644, 0.778, 0.022, 0.983, 0.926]
  Student vector: [0.089, 0.111, 0.044, 0.906, 0.852]
  |differences|:  [0.556, 0.667, 0.022, 0.078, 0.074]
  distance       = 0.2793
  F(x)           = 0.7207

SUMMARY TABLE
  Level  Distance    F(x)
      1    0.2793  0.7207
      2    0.1567  0.8433
      3    0.1013  0.8987
      4    0.1427  0.8573
Overall    0.1366  0.8634
```

The upward trend from Level 1 to Level 3 is the problem. F(x) should go down if reasoning fidelity degrades with difficulty, but the length effect on V1 and V2 pushes it the wrong way.

## Where this fits in the pipeline

1. `run_experiment.py` generates model outputs
2. `rescore.py` corrects answer extraction
3. `extract_features.py` extracts V1-V5 features
4. **`compute_fidelity.py` computes the composite fidelity score** (you are here)
5. The fidelity score was abandoned; `h3_regression.py` replaced it with a logistic regression on accuracy

This script is included for completeness. The dissertation discusses its failure in Section 3.1.6 ("Note on the composite fidelity score") and in Section 4.4 ("Methodological reflections").
