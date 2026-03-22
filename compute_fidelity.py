"""
compute_fidelity.py
-------------------
Computes the reasoning fidelity score F(x) per complexity level.

F(x) = 1 - distance(v_T(x), v_S(x))

where distance is the mean absolute difference across V1-V5,
and v_T / v_S are the level-mean feature vectors for teacher / student.

Usage:
    python3 compute_fidelity.py --teacher teacher_features.jsonl --student student_features.jsonl
"""

import argparse
import numpy as np
import pandas as pd

FEATURES = ['V1', 'V2', 'V3', 'V4', 'V5']


def load_features(path):
    """Load a features file (CSV format)."""
    df = pd.read_csv(path)

    # Check required columns exist
    required = ['problem_id', 'complexity_level'] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nERROR: Missing columns in {path}: {missing}")
        print(f"Available columns: {list(df.columns)}")
        exit(1)

    return df


def compute_fidelity(teacher_df, student_df):
    """Compute F(x) per complexity level."""

    # Step 1: Average each feature across runs, per problem
    teacher_avg = teacher_df.groupby('problem_id')[FEATURES].mean()
    student_avg = student_df.groupby('problem_id')[FEATURES].mean()

    # Get level mapping from problem_id
    level_map = teacher_df.groupby('problem_id')['complexity_level'].first()
    teacher_avg['level'] = level_map
    student_avg['level'] = level_map

    # Step 2: Average across problems at each level
    teacher_level = teacher_avg.groupby('level')[FEATURES].mean()
    student_level = student_avg.groupby('level')[FEATURES].mean()

    # Step 3: Compute F(x) per level
    print("=" * 65)
    print("FIDELITY SCORE COMPUTATION: F(x) = 1 - mean(|v_T - v_S|)")
    print("=" * 65)

    results = []
    for level in sorted(teacher_level.index):
        t_vec = teacher_level.loc[level].values
        s_vec = student_level.loc[level].values
        abs_diffs = np.abs(t_vec - s_vec)
        dist = np.mean(abs_diffs)
        fidelity = 1.0 - dist

        print(f"\n--- Level {level} ---")
        print(f"  Teacher vector: [{', '.join(f'{v:.3f}' for v in t_vec)}]")
        print(f"  Student vector: [{', '.join(f'{v:.3f}' for v in s_vec)}]")
        print(f"  |differences|:  [{', '.join(f'{v:.3f}' for v in abs_diffs)}]")
        print(f"  distance       = {dist:.4f}")
        print(f"  F(x)           = {fidelity:.4f}")

        results.append({
            'Level': int(level),
            'T_V1': round(t_vec[0], 3), 'T_V2': round(t_vec[1], 3),
            'T_V3': round(t_vec[2], 3), 'T_V4': round(t_vec[3], 3),
            'T_V5': round(t_vec[4], 3),
            'S_V1': round(s_vec[0], 3), 'S_V2': round(s_vec[1], 3),
            'S_V3': round(s_vec[2], 3), 'S_V4': round(s_vec[3], 3),
            'S_V5': round(s_vec[4], 3),
            'Distance': round(dist, 4),
            'F(x)': round(fidelity, 4),
        })

    # Overall F(x)
    t_overall = teacher_df[FEATURES].mean().values
    s_overall = student_df[FEATURES].mean().values
    overall_dist = np.mean(np.abs(t_overall - s_overall))
    overall_f = 1.0 - overall_dist

    print(f"\n--- Overall ---")
    print(f"  Teacher vector: [{', '.join(f'{v:.3f}' for v in t_overall)}]")
    print(f"  Student vector: [{', '.join(f'{v:.3f}' for v in s_overall)}]")
    print(f"  distance       = {overall_dist:.4f}")
    print(f"  F(x)           = {overall_f:.4f}")

    results.append({
        'Level': 'Overall',
        'T_V1': round(t_overall[0], 3), 'T_V2': round(t_overall[1], 3),
        'T_V3': round(t_overall[2], 3), 'T_V4': round(t_overall[3], 3),
        'T_V5': round(t_overall[4], 3),
        'S_V1': round(s_overall[0], 3), 'S_V2': round(s_overall[1], 3),
        'S_V3': round(s_overall[2], 3), 'S_V4': round(s_overall[3], 3),
        'S_V5': round(s_overall[4], 3),
        'Distance': round(overall_dist, 4),
        'F(x)': round(overall_f, 4),
    })

    # Summary table
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 65)
    print("SUMMARY TABLE")
    print("=" * 65)
    print(results_df[['Level', 'Distance', 'F(x)']].to_string(index=False))

    # Save to CSV
    results_df.to_csv('fidelity_results.csv', index=False)
    print(f"\nFull results saved to: fidelity_results.csv")

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Compute fidelity score F(x)')
    parser.add_argument('--teacher', required=True, help='Path to teacher features CSV')
    parser.add_argument('--student', required=True, help='Path to student features CSV')
    args = parser.parse_args()

    # Load data
    print(f"Loading teacher features from: {args.teacher}")
    teacher = load_features(args.teacher)
    print(f"  -> {len(teacher)} rows, {teacher['problem_id'].nunique()} problems")

    print(f"Loading student features from: {args.student}")
    student = load_features(args.student)
    print(f"  -> {len(student)} rows, {student['problem_id'].nunique()} problems")

    # Sanity checks
    t_problems = set(teacher['problem_id'].unique())
    s_problems = set(student['problem_id'].unique())
    if t_problems != s_problems:
        only_t = t_problems - s_problems
        only_s = s_problems - t_problems
        if only_t:
            print(f"\nWARNING: {len(only_t)} problems in teacher but not student: {only_t}")
        if only_s:
            print(f"\nWARNING: {len(only_s)} problems in student but not teacher: {only_s}")
        shared = t_problems & s_problems
        teacher = teacher[teacher['problem_id'].isin(shared)]
        student = student[student['problem_id'].isin(shared)]
        print(f"Using {len(shared)} shared problems.")

    # Check feature ranges
    for feat in FEATURES:
        for name, df in [('Teacher', teacher), ('Student', student)]:
            mn, mx = df[feat].min(), df[feat].max()
            if mn < 0 or mx > 1:
                print(f"\nWARNING: {name} {feat} has values outside [0,1]: min={mn}, max={mx}")

    compute_fidelity(teacher, student)


if __name__ == '__main__':
    main()
