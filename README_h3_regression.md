# h3_regression_corrected.py

Runs the logistic regression that tests whether the accuracy gap between teacher and student widens with problem difficulty. This was originally planned as a standalone hypothesis (H3) but was folded into the H1 analysis after the composite fidelity metric failed. The regression itself remained the same.

## What it does

1. Loads the teacher and student feature CSVs (which contain automated correctness scores from `extract_features.py`)
2. Loads manual review Excel sheets where I corrected answers that the automated scorer got wrong
3. Overrides the automated `correct` column with the manual verdict wherever one exists
4. Combines the teacher and student data into a single dataframe (400 rows: 200 per model)
5. Fits a logistic regression predicting correctness from three variables: whether the model is the student, the complexity level (1-4), and the interaction between the two
6. Prints the full regression table, accuracy gaps by level, odds ratios, and model fit statistics
7. Saves the regression coefficients to a CSV

## Why the manual correction step matters

The automated scorer from `run_experiment.py` missed correct answers due to formatting issues (LaTeX mismatches, answers wrapped in sentences, etc.). `rescore.py` caught most of these, but some still needed a human to read the full output and judge. The manual review Excel sheets contain those human judgements in a `YOUR_VERDICT` column (Y or N). This script applies those corrections before running the regression, so the results use the final corrected accuracy numbers reported in the dissertation.

## The regression model

```
logit(P(correct)) = b0 + b1 * is_student + b2 * complexity + b3 * (is_student * complexity)
```

The coefficient that matters is b3, the interaction term. If b3 is negative and significant, it means the student falls behind the teacher faster on harder problems (the gap widens). If b3 is near zero and not significant, the gap is roughly the same at every difficulty level.

The result: b3 = -0.040, p = 0.877. No evidence the gap widens. Distillation imposes a roughly uniform accuracy penalty of 7-12 percentage points across all four difficulty levels.

## Requirements

- Python 3.8+
- `pandas`, `numpy`, `statsmodels`, `openpyxl`

```
pip install pandas numpy statsmodels openpyxl
```

## Usage

```bash
python3 h3_regression_corrected.py \
    --teacher teacher_features.csv \
    --student student_features.csv \
    --teacher-review teacher_manual_review.xlsx \
    --student-review student_manual_review_FULL.xlsx
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--teacher` | Path to the teacher's feature CSV (from `extract_features.py`) |
| `--student` | Path to the student's feature CSV (from `extract_features.py`) |
| `--teacher-review` | Path to the teacher's manual review Excel file |
| `--student-review` | Path to the student's manual review Excel file |

## Input file format

The feature CSVs need at minimum: `problem_id`, `model_name`, `run_id`, `correct`, `complexity_level`.

The manual review Excel files need: `problem_id` in column A, `run_id` in column B, and `YOUR_VERDICT` (Y or N) in column G. These were produced by `generate_validation_sheet.py` and filled in by hand.

## Output

The script prints the full `statsmodels` regression summary to the terminal, plus a condensed version with the key coefficients, p-values, odds ratios, and an interpretation in plain English.

It also saves `h3_regression_corrected.csv` containing one row per coefficient with columns for the estimate, standard error, z-value, p-value, 95% confidence interval bounds, and odds ratio. This CSV is the source for Table 7 in the dissertation.

## Example terminal output (condensed)

```
KEY RESULTS (CORRECTED)
=================================================================

Intercept (b0):              +3.7740  (p=0.0000)
is_student (b1):             -0.3990  (p=0.5856)
complexity (b2):             -1.2680  (p=0.0000)
is_student x complexity (b3):-0.0400  (p=0.8770)

--- Interpretation ---
b3 is NOT SIGNIFICANT (p=0.8770).
-> No evidence the accuracy gap widens with complexity.
   H3 is not supported at the 0.05 level.
```

## Where this fits in the pipeline

1. `run_experiment.py` generates model outputs and initial scores
2. `rescore.py` corrects answer extraction
3. `extract_features.py` extracts V1-V5 features (and carries through the `correct` column)
4. Manual review in Excel corrects remaining scoring errors
5. **`h3_regression_corrected.py` applies the manual corrections and runs the regression** (you are here)

## Note on naming

The script and output file are called "h3" because the regression was originally planned as a standalone third hypothesis. In the final dissertation, this analysis is reported within the H1 results section (Section 3.1.5, "Interaction between model and complexity") rather than as a separate hypothesis. The script was not renamed to avoid breaking file references.
