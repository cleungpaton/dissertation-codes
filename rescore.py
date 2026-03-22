"""
=============================================================================
Re-score existing experiment results with improved answer matching
=============================================================================
Reads an existing run_log.jsonl, re-extracts and re-scores all answers using
the improved extraction logic, and produces corrected output files.

Does NOT call Ollama - just re-processes existing data. Takes seconds.

Usage:
  python3 rescore.py --input results_student/run_log.jsonl --output results_student_rescored/
=============================================================================
"""

import json
import re
import os
import pandas as pd
import argparse
from datetime import datetime


# =============================================================================
# IMPROVED ANSWER EXTRACTION (handles nested braces, dollar signs, text wrappers)
# =============================================================================

def extract_final_answer(output: str) -> str:
    lines = output.strip().splitlines()
    
    # 1. Explicit FINAL ANSWER line
    for line in reversed(lines):
        cleaned = line.strip()
        if cleaned.upper().startswith("FINAL ANSWER:"):
            answer = cleaned.split(":", 1)[1].strip()
            return clean_extracted_answer(answer)
    
    # 2. \boxed{...} with balanced brace handling
    boxed_answers = []
    for match in re.finditer(r'\\boxed\{', output):
        start = match.end()
        depth = 1
        pos = start
        while pos < len(output) and depth > 0:
            if output[pos] == '{':
                depth += 1
            elif output[pos] == '}':
                depth -= 1
            pos += 1
        if depth == 0:
            boxed_answers.append(output[start:pos-1].strip())
    
    if boxed_answers:
        return clean_extracted_answer(boxed_answers[-1])
    
    # 3. Last non-empty line
    for line in reversed(lines):
        if line.strip():
            return clean_extracted_answer(line.strip())
    
    return ""


def clean_extracted_answer(answer: str) -> str:
    s = answer.strip()
    
    # Remove bold/italic markers
    s = s.replace("**", "").replace("__", "")
    
    # Handle escaped dollar signs and currency
    s = s.replace("\\$", "$")
    s = re.sub(r'^\$\s*', '', s)
    s = re.sub(r'\s*\$$', '', s)
    
    # If it looks like a sentence with a number, extract the number
    if len(s) > 20:
        text_match = re.search(
            r'(?:answer|paid|costs?|earns?|needs?|removes?|is|equals?)[:\s]*'
            r'(?:.*?)(\$?\s*-?[\d,]+(?:\.\d+)?)',
            s, re.IGNORECASE
        )
        if text_match:
            extracted_num = text_match.group(1).replace("$", "").replace(",", "").strip()
            if extracted_num:
                s = extracted_num
    
    # Remove trailing text descriptions
    s = re.sub(r'\s*(more expensive|plates|for the shoes|clips|dollars|hours)\.?\s*$', 
               '', s, flags=re.IGNORECASE)
    s = s.rstrip(".")
    
    return s.strip()


# =============================================================================
# IMPROVED NORMALISATION
# =============================================================================

def normalise_answer(answer: str) -> str:
    s = str(answer).strip()
    
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "")
    s = s.strip(".")
    
    # Remove \boxed{...} wrapper
    boxed_match = re.search(r'\\boxed\{(.+)\}$', s)
    if boxed_match:
        s = boxed_match.group(1)
    
    s = " ".join(s.split())
    
    # Normalise exponents
    s = re.sub(r'\*\*', '^', s)
    s = re.sub(r'\^{(\d+)}', r'^\1', s)
    s = re.sub(r'\^\{([^}]+)\}', r'^(\1)', s)
    
    # Normalise LaTeX
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    s = s.replace("\\text{", "").replace("}", "")
    
    # Normalise fractions
    s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', s)
    s = re.sub(r'\\dfrac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', s)
    
    # Normalise sqrt
    s = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', s)
    
    # Remove remaining LaTeX backslash commands
    s = re.sub(r'\\([a-zA-Z]+)', r'\1', s)
    
    # Normalise comma-separated large numbers (1,000 -> 1000)
    s = re.sub(r'(\d),(\d{3})(?!\d)', r'\1\2', s)
    
    return s.strip().lower()


def check_answer(model_answer: str, gold_answer: str) -> dict:
    norm_model = normalise_answer(model_answer)
    norm_gold = normalise_answer(gold_answer)
    
    # Level 1: exact string
    if norm_model == norm_gold:
        return {"correct": 1, "method": "exact_string"}
    
    # Also try without spaces (catches "x^2 + 7x + 10" vs "x^2+7x+10")
    if norm_model.replace(" ", "") == norm_gold.replace(" ", ""):
        return {"correct": 1, "method": "exact_nospace"}
    
    # Level 2: numeric
    try:
        def extract_number(s):
            frac_match = re.match(r'^(-?\d+)\s*/\s*(\d+)$', s.strip())
            if frac_match:
                return float(frac_match.group(1)) / float(frac_match.group(2))
            num_match = re.match(r'^-?[\d.]+$', s.strip())
            if num_match:
                return float(s.strip())
            return None
        
        model_num = extract_number(norm_model)
        gold_num = extract_number(norm_gold)
        
        if model_num is not None and gold_num is not None:
            if abs(model_num - gold_num) < 1e-6:
                return {"correct": 1, "method": "numeric"}
    except:
        pass
    
    # Level 3: symbolic
    try:
        from sympy.parsing.sympy_parser import parse_expr
        from sympy import simplify
        
        model_expr = parse_expr(norm_model.replace("^", "**"))
        gold_expr = parse_expr(norm_gold.replace("^", "**"))
        
        if simplify(model_expr - gold_expr) == 0:
            return {"correct": 1, "method": "symbolic"}
    except:
        pass
    
    return {"correct": 0, "method": "no_match"}


# =============================================================================
# MAIN
# =============================================================================

def rescore(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading: {input_path}")
    
    entries = []
    with open(input_path) as f:
        for line in f:
            entries.append(json.loads(line))
    
    print(f"Total entries: {len(entries)}")
    
    # Re-extract and re-score each entry
    changes = 0
    for entry in entries:
        old_correct = entry["correct"]
        old_extracted = entry["extracted_answer"]
        
        # Re-extract answer from full output
        new_extracted = extract_final_answer(entry["full_output"])
        new_score = check_answer(new_extracted, entry["gold_answer"])
        
        entry["extracted_answer"] = new_extracted
        entry["correct"] = new_score["correct"]
        entry["match_method"] = new_score["method"]
        
        if old_correct != new_score["correct"]:
            changes += 1
            direction = "✗→✓" if new_score["correct"] == 1 else "✓→✗"
            print(f"  {direction} {entry['problem_id']} run {entry['run_id']}: "
                  f"'{old_extracted}' → '{new_extracted}' "
                  f"(gold: '{entry['gold_answer']}') [{new_score['method']}]")
    
    print(f"\nTotal scoring changes: {changes}")
    
    # Save corrected JSONL
    corrected_log = os.path.join(output_dir, "run_log_rescored.jsonl")
    with open(corrected_log, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved: {corrected_log}")
    
    # Build summary DataFrames
    df = pd.DataFrame(entries)
    
    # Summary CSV
    cols_for_csv = [c for c in df.columns if c != "full_output"]
    summary_path = os.path.join(output_dir, "summary_rescored.csv")
    df[cols_for_csv].to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    
    # Accuracy by complexity
    if "complexity_level" in df.columns:
        acc = df.groupby(["model_name", "complexity_level"]).agg(
            total=("correct", "count"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
        ).round(3)
        acc_path = os.path.join(output_dir, "accuracy_by_complexity.csv")
        acc.to_csv(acc_path)
        print(f"\nAccuracy by complexity (RESCORED):")
        print(acc.to_string())
    
    # Overall accuracy
    overall = df.groupby("model_name").agg(
        total=("correct", "count"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
    ).round(3)
    overall_path = os.path.join(output_dir, "accuracy_overall.csv")
    overall.to_csv(overall_path)
    print(f"\nOverall accuracy (RESCORED):")
    print(overall.to_string())
    
    # Manual review file
    no_match = df[df["match_method"] == "no_match"]
    if len(no_match) > 0:
        review_path = os.path.join(output_dir, "needs_manual_review.csv")
        review_cols = ["problem_id", "model_name", "run_id",
                       "extracted_answer", "gold_answer"]
        if "complexity_level" in no_match.columns:
            review_cols.append("complexity_level")
        no_match[review_cols].to_csv(review_path, index=False)
        print(f"\n⚠ {len(no_match)} answers still need manual review: {review_path}")
    
    print(f"\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-score experiment results")
    parser.add_argument("--input", required=True, help="Path to run_log.jsonl")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    rescore(args.input, args.output)
