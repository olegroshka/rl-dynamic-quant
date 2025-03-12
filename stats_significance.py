#!/usr/bin/env python
"""
stats_significance.py

Compute statistical significance and relative error reduction
given two arrays of accuracies from repeated experiments.

Usage example:

  python stats_significance.py --acc_a 0.407,0.409,0.405 --acc_b 0.402,0.406,0.408

Or pass them via a file or any other approach.
"""
import argparse
import math
import statistics
from typing import List
import sys

import numpy as np
from scipy.stats import ttest_rel  # Paired t-test


def relative_error_reduction(acc_a: float, acc_b: float) -> float:
    """
    Given two average accuracies, acc_a and acc_b,
    compute the relative error reduction of moving from B -> A.
    If "b" is baseline with accuracy acc_b, error_b = 1 - acc_b,
    error_a = 1 - acc_a,
    RER = (error_b - error_a) / error_b = 1 - (error_a / error_b).
    """
    error_a = 1.0 - acc_a
    error_b = 1.0 - acc_b
    if error_b <= 0:
        return 0.0  # or float('nan') if baseline is "perfect"
    return (error_b - error_a) / error_b


def sign_test_wins(acc_a_arr: List[float], acc_b_arr: List[float]):
    """
    A simple sign test approach for small sets of repeated measures.
    Count how many times A > B, how many times B > A, ignoring ties.
    This is simpler than t-test, but less commonly used.
    We won't compute a p-value here, just a quick "wins/ties" summary.
    """
    wins_a = 0
    wins_b = 0
    ties = 0
    for a, b in zip(acc_a_arr, acc_b_arr):
        if abs(a - b) < 1e-9:
            ties += 1
        elif a > b:
            wins_a += 1
        else:
            wins_b += 1
    return wins_a, wins_b, ties


def main():
    parser = argparse.ArgumentParser(description="Statistical significance for two sets of accuracies.")
    parser.add_argument("--acc_a", type=str, required=True,
                        help="Comma-separated accuracies for method A, e.g. '0.407,0.409,0.405'")
    parser.add_argument("--acc_b", type=str, required=True,
                        help="Comma-separated accuracies for method B, e.g. '0.402,0.406,0.408'")
    parser.add_argument("--label_a", type=str, default="Method A")
    parser.add_argument("--label_b", type=str, default="Method B")

    args = parser.parse_args()

    # Parse the accuracy strings into float lists
    acc_a_arr = [float(x) for x in args.acc_a.split(",")]
    acc_b_arr = [float(x) for x in args.acc_b.split(",")]

    if len(acc_a_arr) != len(acc_b_arr):
        print("Error: Arrays must be of the same length!")
        sys.exit(1)

    # Basic stats
    mean_a = statistics.mean(acc_a_arr)
    mean_b = statistics.mean(acc_b_arr)
    std_a = statistics.pstdev(acc_a_arr)  # or stdev for sample
    std_b = statistics.pstdev(acc_b_arr)

    # T-test (paired)
    t_stat, p_val = ttest_rel(acc_a_arr, acc_b_arr)

    # Relative error reduction
    rer = relative_error_reduction(mean_a, mean_b)

    # Sign test
    wins_a, wins_b, ties = sign_test_wins(acc_a_arr, acc_b_arr)

    print(f"== {args.label_a} vs {args.label_b} ==")
    print(f"{args.label_a}: mean={mean_a:.4f}, std={std_a:.4f}")
    print(f"{args.label_b}: mean={mean_b:.4f}, std={std_b:.4f}")
    print(f"Absolute diff: {mean_a - mean_b:.4f}")
    print(f"Relative error reduction: {rer*100:.2f}%")
    print(f"Paired t-test: t={t_stat:.4f}, p={p_val:.4f}")
    print(f"Sign test summary: wins_A={wins_a}, wins_B={wins_b}, ties={ties}")


if __name__ == "__main__":
    main()
