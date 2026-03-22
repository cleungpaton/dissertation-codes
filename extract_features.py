"""
=============================================================================
Feature Extraction for Hypothesis 1: Verification Behaviour Analysis
=============================================================================
Extracts the V1-V5 reasoning feature vector from each model output.

Features:
  V1 = Verification present (binary) - did the model check its own work?
  V2 = Restart/hesitation markers (binary) - signs of reasoning instability
  V3 = Contradiction markers (binary) - logical inconsistencies present
  V4 = Symbolic engagement (count / 4) - number of explicit equations, capped
  V5 = Solution structure score (0-1) - how organised is the reasoning

Usage:
  python extract_features.py --input results/run_log.jsonl --output results/features.csv

This produces a CSV with one row per (problem_id, model_name, run_id) and
columns V1-V5 plus the raw indicators that led to each score.
=============================================================================
"""

import json
import re
import pandas as pd
import argparse
from pathlib import Path


# =============================================================================
# V1: VERIFICATION BEHAVIOUR
# =============================================================================

# Keywords/patterns that indicate the model is checking its own work
VERIFICATION_PATTERNS = [
    # Explicit verification language
    r"let.{0,5}s?\s*(verify|check|confirm|validate)",
    r"(verif|check)(y|ing|ied)\s+(this|the|our|my|that)",
    r"substitut(e|ing).{0,30}(back|into|original)",
    r"plug(ging)?\s+(back|in|into)",
    # Domain/condition checking
    r"(check|verify|ensure|confirm).{0,20}(domain|condition|restriction|constraint)",
    r"(domain|range)\s+(is|restriction|requires)",
    r"(must\s+be|requires?|need)\s+(positive|non-?negative|non-?zero|greater|defined)",
    # Extraneous root checking
    r"extraneous\s+(root|solution)",
    r"(discard|reject|exclude|invalid)\s+(this\s+)?(root|solution|answer|value)",
    # General checking
    r"(sanity|sense)\s+check",
    r"(does|this)\s+(make|seem)\s+(sense|right|correct)",
    r"(double|cross).?check",
    r"to\s+confirm",
    r"we\s+can\s+verify",
]

def extract_v1(text: str) -> dict:
    """V1: Is verification behaviour present?"""
    text_lower = text.lower()
    matches = []
    for pattern in VERIFICATION_PATTERNS:
        found = re.findall(pattern, text_lower)
        if found:
            matches.append(pattern)
    
    return {
        "V1": 1 if len(matches) > 0 else 0,
        "V1_match_count": len(matches),
        "V1_patterns": "; ".join(matches[:5]),  # store first 5 for debugging
    }


# =============================================================================
# V2: RESTART / HESITATION MARKERS
# =============================================================================

HESITATION_PATTERNS = [
    r"wait,?\s",
    r"actually,?\s",
    r"no,?\s+(that|this|wait|let)",
    r"let\s+me\s+(try|re|start|think)\s+(a\s+different|again|over|about)",
    r"(that.s|that\s+is)\s+(not\s+right|wrong|incorrect|a\s+mistake)",
    r"I\s+(made|think\s+I\s+made)\s+a\s+(mistake|error)",
    r"(scratch|scrap)\s+that",
    r"(going|go)\s+back\s+to",
    r"(start|begin)(ing)?\s+(over|again|fresh)",
    r"hmm",
    r"oops",
    r"correction:",
]

def extract_v2(text: str) -> dict:
    """V2: Are restart/hesitation markers present?"""
    text_lower = text.lower()
    matches = []
    for pattern in HESITATION_PATTERNS:
        found = re.findall(pattern, text_lower)
        if found:
            matches.append(pattern)
    
    return {
        "V2": 1 if len(matches) > 0 else 0,
        "V2_match_count": len(matches),
        "V2_patterns": "; ".join(matches[:5]),
    }


# =============================================================================
# V3: CONTRADICTION MARKERS
# =============================================================================

CONTRADICTION_PATTERNS = [
    r"(but\s+)?(earlier|above|before)\s+(we|I)\s+(said|found|showed|assumed)\s",
    r"(this\s+)?contradicts?\s",
    r"(inconsisten|contradict)",
    r"(but|however)\s+(this|that)\s+(means|implies|gives|would)",
    # Assuming something then violating it
    r"(assum(e|ed|ing)\s+.{0,40}(but|however|yet))",
]

def extract_v3(text: str) -> dict:
    """V3: Are contradiction markers present?"""
    text_lower = text.lower()
    matches = []
    for pattern in CONTRADICTION_PATTERNS:
        found = re.findall(pattern, text_lower)
        if found:
            matches.append(pattern)
    
    return {
        "V3": 1 if len(matches) > 0 else 0,
        "V3_match_count": len(matches),
        "V3_patterns": "; ".join(matches[:5]),
    }


# =============================================================================
# V4: SYMBOLIC ENGAGEMENT
# =============================================================================

def extract_v4(text: str) -> dict:
    """
    V4: Level of symbolic/mathematical engagement.
    Count explicit equations or mathematical expressions, capped at 4.
    Score = min(count, 4) / 4
    """
    # Count lines containing equals signs with mathematical content
    equation_patterns = [
        r'[a-zA-Z0-9\)]\s*=\s*[a-zA-Z0-9\(]',  # x = 5, f(x) = ...
        r'\\frac\{',                                # LaTeX fractions
        r'\\sqrt\{',                                # LaTeX roots
        r'\\int',                                    # integrals
        r'\\sum',                                    # summations
        r'\^{?\d',                                   # exponents
        r'\d+\s*[\+\-\*/]\s*\d+\s*=',              # arithmetic: 3 + 5 = 8
    ]
    
    equation_count = 0
    for line in text.split("\n"):
        for pattern in equation_patterns:
            if re.search(pattern, line):
                equation_count += 1
                break  # count each line only once
    
    capped = min(equation_count, 4)
    
    return {
        "V4": round(capped / 4, 2),
        "V4_raw_count": equation_count,
        "V4_capped": capped,
    }


# =============================================================================
# V5: SOLUTION STRUCTURE
# =============================================================================

def extract_v5(text: str) -> dict:
    """
    V5: How structured/organised is the solution?
    Composite score based on:
      - Has clear step separation (numbered steps, "Step 1", etc.)
      - Has a clearly marked conclusion/answer
      - Reasonable length (not just a one-liner, not absurdly long)
    Score: average of 3 binary indicators -> [0, 0.33, 0.67, 1.0]
    """
    text_lower = text.lower()
    
    # Check for step markers
    step_patterns = [
        r'step\s*\d',
        r'^\s*\d+[\.\)]\s',     # "1. " or "1) "
        r'(first|second|third|next|then|finally)',
    ]
    has_steps = 0
    for p in step_patterns:
        if re.search(p, text_lower, re.MULTILINE):
            has_steps = 1
            break
    
    # Check for clear conclusion
    conclusion_patterns = [
        r'(therefore|thus|hence|so|the\s+answer\s+is|final\s+answer)',
        r'\\boxed\{',
        r'FINAL ANSWER:',
    ]
    has_conclusion = 0
    for p in conclusion_patterns:
        if re.search(p, text, re.IGNORECASE):
            has_conclusion = 1
            break
    
    # Check reasonable length (between 50 and 5000 chars for a maths solution)
    reasonable_length = 1 if 50 < len(text) < 5000 else 0
    
    score = round((has_steps + has_conclusion + reasonable_length) / 3, 2)
    
    return {
        "V5": score,
        "V5_has_steps": has_steps,
        "V5_has_conclusion": has_conclusion,
        "V5_reasonable_length": reasonable_length,
        "V5_char_count": len(text),
    }


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def extract_all_features(text: str) -> dict:
    """Extract all V1-V5 features from a single model output."""
    features = {}
    features.update(extract_v1(text))
    features.update(extract_v2(text))
    features.update(extract_v3(text))
    features.update(extract_v4(text))
    features.update(extract_v5(text))
    return features


def process_log(input_path: str, output_path: str):
    """
    Read the JSONL run log and extract features for every output.
    """
    print(f"Reading: {input_path}")
    
    rows = []
    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            entry = json.loads(line)
            
            # Extract features from the full output
            features = extract_all_features(entry.get("full_output", ""))
            
            # Combine metadata + features
            row = {
                "problem_id": entry["problem_id"],
                "model_name": entry["model_name"],
                "run_id": entry["run_id"],
                "correct": entry["correct"],
                "complexity_level": entry.get("complexity_level", ""),
                "domain": entry.get("domain", ""),
                "dataset": entry.get("dataset", ""),
            }
            row.update(features)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(df)} rows)")
    
    # Print summary
    print("\n--- Feature Means by Model ---")
    feature_cols = ["V1", "V2", "V3", "V4", "V5"]
    summary = df.groupby("model_name")[feature_cols].mean().round(3)
    print(summary.to_string())
    
    if "complexity_level" in df.columns:
        print("\n--- V1 (Verification) by Model and Complexity ---")
        v1_summary = df.groupby(["model_name", "complexity_level"])["V1"].mean().round(3)
        print(v1_summary.to_string())
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract V1-V5 reasoning features from experiment log"
    )
    parser.add_argument("--input", required=True,
                        help="Path to run_log.jsonl from the experiment")
    parser.add_argument("--output", default="results/features.csv",
                        help="Output CSV path")
    
    args = parser.parse_args()
    process_log(args.input, args.output)
