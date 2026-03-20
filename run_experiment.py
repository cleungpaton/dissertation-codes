"""
=============================================================================
LLM Mathematical Reasoning Fidelity Pipeline
=============================================================================
MATH3001 Dissertation - Knowledge Distillation & Reasoning Fidelity

Models:
  Teacher : hengwen/DeepSeek-R1-Distill-Qwen-32B:q4_k_m
  Student : erwan2/DeepSeek-R1-Distill-Qwen-1.5B  (or deepseek-r1:1.5b)
  Baseline: qwen2.5:1.5b  (for H2 only - add later)

Usage:
  python run_experiment.py --input questions.csv --output results/ --runs 5
  python run_experiment.py --input questions.csv --output results/ --runs 5 --models student
  python run_experiment.py --input questions.csv --output results/ --runs 1 --limit 5  # quick test

Prerequisites:
  pip install pandas requests openpyxl sympy --break-system-packages
  Ollama running locally at http://localhost:11434
=============================================================================
"""

import pandas as pd
import requests
import json
import time
import os
import argparse
import hashlib
from datetime import datetime
from pathlib import Path


# =============================================================================
# CONFIGURATION - edit these to match your setup
# =============================================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = {
    "teacher": "hengwen/DeepSeek-R1-Distill-Qwen-32B:q4_k_m",
    "student": "erwan2/DeepSeek-R1-Distill-Qwen-1.5B",
    # Uncomment when ready for H2:
    "baseline": "qwen2.5:1.5b",
}

# Decoding parameters - DOCUMENT THESE IN YOUR REPORT
TEMPERATURE = 0.6       # >0 needed for repeated runs to show variation
TOP_P = 0.95            # nucleus sampling
MAX_TOKENS = 4096       # enough for full chain-of-thought
SEED = None             # set to an int for reproducibility within a single run

PROMPT_VERSION = "v1_cot_final"


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def build_prompt(question: str) -> str:
    """
    Prompt that elicits full chain-of-thought reasoning AND a clearly
    demarcated final answer. This serves all three hypotheses:
      - H1: full reasoning trace lets you extract V1-V5 verification features
      - H2: full trace lets you classify error types
      - H3: accuracy + fidelity scored per complexity level
    
    IMPORTANT: We do NOT use few-shot examples because:
      1. It keeps the prompt identical for both models (no bias from examples)
      2. DeepSeek-R1 distilled models already have chain-of-thought built in
      3. Simpler prompt = more reproducible = easier to defend in your report
    """
    return (
        "Solve the following mathematics problem. "
        "Show your complete working and reasoning step by step. "
        "After your working, write your final answer on a new line "
        "in exactly this format:\n"
        "FINAL ANSWER: <your answer>\n\n"
        f"Problem: {question}"
    )


# =============================================================================
# OLLAMA API CALL
# =============================================================================

def call_ollama(model: str, prompt: str, timeout: int = 600) -> dict:
    """
    Call the Ollama API and return the full response including timing.
    
    Returns dict with:
      - response: the model's text output
      - total_duration: total time in nanoseconds (from Ollama)
      - eval_count: number of tokens generated
      - error: error message if something went wrong
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_predict": MAX_TOKENS,
        }
    }
    if SEED is not None:
        payload["options"]["seed"] = SEED

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return {
            "response": data.get("response", "").strip(),
            "total_duration": data.get("total_duration", 0),
            "eval_count": data.get("eval_count", 0),
            "error": None,
        }
    except requests.exceptions.Timeout:
        return {"response": "", "total_duration": 0, "eval_count": 0,
                "error": "TIMEOUT"}
    except requests.exceptions.ConnectionError:
        return {"response": "", "total_duration": 0, "eval_count": 0,
                "error": "CONNECTION_ERROR - is Ollama running?"}
    except Exception as e:
        return {"response": "", "total_duration": 0, "eval_count": 0,
                "error": str(e)}


# =============================================================================
# ANSWER EXTRACTION AND MATCHING
# =============================================================================

def extract_final_answer(output: str) -> str:
    """
    Extract the final answer from model output.
    Priority order:
      1. Explicit 'FINAL ANSWER:' line
      2. \\boxed{...} (with nested brace handling)
      3. Last non-empty line (fallback)
    """
    import re
    lines = output.strip().splitlines()
    
    # Try to find explicit FINAL ANSWER line
    for line in reversed(lines):
        cleaned = line.strip()
        if cleaned.upper().startswith("FINAL ANSWER:"):
            answer = cleaned.split(":", 1)[1].strip()
            return clean_extracted_answer(answer)
    
    # Fallback: look for boxed answers with nested brace handling
    # Find all \boxed{ positions and extract with balanced braces
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
    
    # Last resort: return the last non-empty line
    for line in reversed(lines):
        if line.strip():
            return clean_extracted_answer(line.strip())
    
    return ""


def clean_extracted_answer(answer: str) -> str:
    """
    Clean up common formatting issues in extracted answers.
    Strips dollar signs, bold markers, text wrappers, etc.
    """
    import re
    s = answer.strip()
    
    # Remove bold/italic markers
    s = s.replace("**", "").replace("__", "")
    
    # Remove dollar signs (currency, not LaTeX)
    s = s.replace("\\$", "$")
    s = re.sub(r'^\$\s*', '', s)
    s = re.sub(r'\s*\$$', '', s)
    
    # Remove common text wrappers like "Answer: Alexis paid $41 for the shoes."
    # Pattern: extract the number from sentences containing the answer
    text_match = re.search(
        r'(?:answer|paid|costs?|earns?|needs?|removes?|is|equals?)[:\s]*'
        r'(?:.*?)(\$?\s*-?[\d,]+(?:\.\d+)?)',
        s, re.IGNORECASE
    )
    if text_match and len(s) > 20:  # only if it looks like a sentence
        extracted_num = text_match.group(1).replace("$", "").replace(",", "").strip()
        if extracted_num:
            s = extracted_num
    
    # Remove trailing periods and common suffixes
    s = re.sub(r'\s*(more expensive|plates|for the shoes)\.?\s*$', '', s, flags=re.IGNORECASE)
    s = s.rstrip(".")
    
    return s.strip()


def normalise_answer(answer: str) -> str:
    """
    Normalise an answer string for comparison.
    Handles common variations in mathematical notation.
    """
    import re
    
    s = str(answer).strip()
    
    # Remove common wrappers
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "")
    s = s.strip(".")
    
    # Remove \\boxed{...} wrapper if present (handles nested braces)
    boxed_match = re.search(r'\\boxed\{(.+)\}$', s)
    if boxed_match:
        s = boxed_match.group(1)
    
    # Normalise whitespace
    s = " ".join(s.split())
    
    # Normalise exponent notation: 2^1009, 2^{1009}, 2**1009 -> 2^1009
    s = re.sub(r'\*\*', '^', s)                    # 2**1009 -> 2^1009
    s = re.sub(r'\^{(\d+)}', r'^\1', s)            # 2^{1009} -> 2^1009
    s = re.sub(r'\^\{([^}]+)\}', r'^(\1)', s)      # x^{n+1} -> x^(n+1)
    
    # Normalise common LaTeX commands
    s = s.replace("\\cdot", "*")
    s = s.replace("\\times", "*")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    s = s.replace("\\text{", "").replace("}", "")
    
    # Normalise fractions: \frac{a}{b} -> a/b
    s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', s)
    s = re.sub(r'\\dfrac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', s)
    
    # Normalise sqrt: \sqrt{x} -> sqrt(x)
    s = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', s)
    
    # Remove remaining backslashes from LaTeX commands
    s = re.sub(r'\\([a-zA-Z]+)', r'\1', s)
    
    # Normalise comma-separated numbers: 1,000 -> 1000 (but not (1,2))
    s = re.sub(r'(\d),(\d{3})(?!\d)', r'\1\2', s)
    
    return s.strip().lower()


def check_answer(model_answer: str, gold_answer: str) -> dict:
    """
    Check if model answer matches gold answer.
    Returns dict with match result and method used.
    
    Three levels of matching:
      1. Exact string match after normalisation
      2. Numeric match (parse both as numbers, compare)
      3. Symbolic match via SymPy (if installed)
    """
    norm_model = normalise_answer(model_answer)
    norm_gold = normalise_answer(gold_answer)
    
    # Level 1: Exact string match
    if norm_model == norm_gold:
        return {"correct": 1, "method": "exact_string"}
    
    # Level 2: Numeric match
    try:
        import re
        # Try to extract numbers
        def extract_number(s):
            # Handle fractions like "1/2"
            frac_match = re.match(r'^(-?\d+)\s*/\s*(\d+)$', s.strip())
            if frac_match:
                return float(frac_match.group(1)) / float(frac_match.group(2))
            # Handle decimals
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
    
    # Level 3: SymPy symbolic match (best effort)
    try:
        from sympy.parsing.sympy_parser import parse_expr
        from sympy import simplify, nsimplify
        
        model_expr = parse_expr(norm_model.replace("^", "**"))
        gold_expr = parse_expr(norm_gold.replace("^", "**"))
        
        if simplify(model_expr - gold_expr) == 0:
            return {"correct": 1, "method": "symbolic"}
    except:
        pass
    
    # No match found - flag for manual review
    return {"correct": 0, "method": "no_match"}


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_experiment(input_path: str, output_dir: str, num_runs: int = 5,
                   model_filter: str = "all", limit: int = None):
    """
    Main experiment loop.
    
    For each problem, for each model, for each run:
      1. Send prompt to model
      2. Store full response (chain-of-thought + answer)
      3. Extract and score the final answer
      4. Log metadata (timing, tokens, etc.)
    
    Output structure:
      results/
        run_log.jsonl          <- every single API call, full detail
        summary.csv            <- one row per (problem, model, run)
        accuracy_summary.csv   <- aggregated accuracy by model/complexity
        experiment_config.json <- full config for reproducibility
    """
    
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load questions
    if input_path.endswith(".xlsx") or input_path.endswith(".xls"):
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)
    
    if limit:
        df = df.head(limit)
        print(f"*** LIMITED TO FIRST {limit} PROBLEMS (test mode) ***")
    
    # Validate required columns
    required = ["problem_id", "question", "final_answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Your columns are: {list(df.columns)}")
        print("\nYour Excel needs at minimum: problem_id, question, final_answer")
        return
    
    # Filter models
    if model_filter == "all":
        active_models = MODELS
    else:
        active_models = {k: v for k, v in MODELS.items() if k in model_filter.split(",")}
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT CONFIGURATION")
    print(f"{'='*60}")
    print(f"Input:       {input_path}")
    print(f"Problems:    {len(df)}")
    print(f"Models:      {list(active_models.keys())}")
    print(f"Runs/model:  {num_runs}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Total calls: {len(df) * len(active_models) * num_runs}")
    print(f"Output:      {output_dir}")
    print(f"{'='*60}\n")
    
    # Save experiment config
    config = {
        "timestamp": datetime.now().isoformat(),
        "input_file": input_path,
        "num_problems": len(df),
        "models": active_models,
        "num_runs": num_runs,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "prompt_version": PROMPT_VERSION,
        "prompt_template": build_prompt("<QUESTION>"),
    }
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Open log file for streaming writes (don't lose data on crash)
    log_path = os.path.join(output_dir, "run_log.jsonl")
    summary_rows = []
    
    total_calls = len(df) * len(active_models) * num_runs
    call_count = 0
    
    with open(log_path, "a") as log_file:
        for run_id in range(1, num_runs + 1):
            for model_name, model_tag in active_models.items():
                for idx, row in df.iterrows():
                    call_count += 1
                    pid = row["problem_id"]
                    question = str(row["question"])
                    gold = str(row["final_answer"])
                    
                    # Progress
                    pct = (call_count / total_calls) * 100
                    print(f"[{call_count}/{total_calls}] ({pct:.0f}%) "
                          f"Run {run_id} | {model_name} | {pid}", end="")
                    
                    # Build prompt and call model
                    prompt = build_prompt(question)
                    t_start = time.time()
                    result = call_ollama(model_tag, prompt)
                    t_elapsed = time.time() - t_start
                    
                    # Extract and score answer
                    extracted = extract_final_answer(result["response"])
                    score = check_answer(extracted, gold)
                    
                    # Build log entry
                    entry = {
                        "problem_id": pid,
                        "model_name": model_name,
                        "model_tag": model_tag,
                        "run_id": run_id,
                        "prompt_version": PROMPT_VERSION,
                        "question": question,
                        "gold_answer": gold,
                        "full_output": result["response"],
                        "extracted_answer": extracted,
                        "correct": score["correct"],
                        "match_method": score["method"],
                        "error": result["error"],
                        "wall_time_s": round(t_elapsed, 2),
                        "ollama_duration_ns": result["total_duration"],
                        "eval_count": result["eval_count"],
                        "timestamp": datetime.now().isoformat(),
                    }
                    
                    # Add metadata columns from the spreadsheet
                    for col in ["dataset", "complexity_level", "domain",
                                "structural_type", "source_split"]:
                        if col in row:
                            entry[col] = row[col]
                    
                    # Write to JSONL (crash-safe: one line per call)
                    log_file.write(json.dumps(entry) + "\n")
                    log_file.flush()
                    
                    # Track for summary
                    summary_rows.append(entry)
                    
                    status = "✓" if score["correct"] else "✗"
                    print(f" | {status} | {t_elapsed:.1f}s")
                    
                    # Small delay to be kind to your machine
                    time.sleep(0.2)
    
    # ==========================================================================
    # POST-PROCESSING: Build summary files
    # ==========================================================================
    
    print(f"\n{'='*60}")
    print("BUILDING SUMMARIES...")
    print(f"{'='*60}\n")
    
    results_df = pd.DataFrame(summary_rows)
    
    # 1. Full summary CSV (one row per problem/model/run)
    summary_path = os.path.join(output_dir, "summary.csv")
    # Don't include full_output in CSV (too large) - it's in the JSONL
    cols_for_csv = [c for c in results_df.columns if c != "full_output"]
    results_df[cols_for_csv].to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    
    # 2. Accuracy summary by model and complexity
    if "complexity_level" in results_df.columns:
        acc = results_df.groupby(["model_name", "complexity_level"]).agg(
            total=("correct", "count"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
        ).round(3)
        acc_path = os.path.join(output_dir, "accuracy_by_complexity.csv")
        acc.to_csv(acc_path)
        print(f"Saved: {acc_path}")
        print("\nAccuracy by model and complexity level:")
        print(acc.to_string())
    
    # 3. Overall accuracy by model
    overall = results_df.groupby("model_name").agg(
        total=("correct", "count"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
    ).round(3)
    overall_path = os.path.join(output_dir, "accuracy_overall.csv")
    overall.to_csv(overall_path)
    print(f"\nSaved: {overall_path}")
    print("\nOverall accuracy:")
    print(overall.to_string())
    
    # 4. Flag manual review cases
    no_match = results_df[results_df["match_method"] == "no_match"]
    if len(no_match) > 0:
        review_path = os.path.join(output_dir, "needs_manual_review.csv")
        review_cols = ["problem_id", "model_name", "run_id",
                       "extracted_answer", "gold_answer"]
        no_match[review_cols].to_csv(review_path, index=False)
        print(f"\n⚠ {len(no_match)} answers need manual review: {review_path}")
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Full output log: {log_path}")
    print(f"{'='*60}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM maths reasoning experiment via Ollama"
    )
    parser.add_argument("--input", required=True,
                        help="Path to CSV or Excel with questions")
    parser.add_argument("--output", default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of repeated runs per model (default: 5)")
    parser.add_argument("--models", default="all",
                        help="Which models to run: all, teacher, student, baseline, "
                             "or comma-separated like 'teacher,student'")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run first N problems (for testing)")
    
    args = parser.parse_args()
    run_experiment(args.input, args.output, args.runs, args.models, args.limit)
