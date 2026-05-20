#!/usr/bin/env python3
"""
Verify cross-fold label CSVs before training.

Checks (SOPInstanceUID = one image; each image has 2 CSV rows: Pectoralis + Nipple):
  1) Pairwise disjoint Test sets across fold 1 (original) and folds 2–10.
  2) Union of all Test sets (across folds) vs full dataset — coverage / partition.
  3) Whether Validation SOP sets are identical across folds (same 200 SOPs) or only
     counts match (stratified size fixed, membership resampled per fold).
  4) Within each fold: Train & Val, Train & Test, Val & Test must be empty.

Setup note (rotating_9 from cv_splits.py):
  Test SOPs are partitioned into 10 disjoint blocks at the image
  (SOPInstanceUID) level whose union covers the 2,000-SOP pool exactly once
  (the defining k-fold property). Validation membership is resampled per
  fold while preserving the original stratified Good/Bad counts; this is a
  fixed-size validation across folds rather than a rotating one.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sop_sets_by_split(df: pd.DataFrame) -> dict[str, set[str]]:
    """Unique SOPs per split (Pectoralis and Nipple rows share SOP → one set suffices)."""
    out: dict[str, set[str]] = {}
    for s in ("Train", "Validation", "Test"):
        out[s] = set(df.loc[df["Split"] == s, "SOPInstanceUID"].astype(str).unique())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify fold label CSVs")
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Repo root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--cv-style",
        choices=("rotating_9", "fixed_test"),
        default="rotating_9",
        help="rotating_9: disjoint tests across fold files (fold 1 + folds 2–10). "
        "fixed_test: all CSVs must share the same Test SOP set.",
    )
    args = parser.parse_args()
    root = args.repo or repo_root()

    paths: dict[int, Path] = {1: root / "labels" / "positioning_labels.csv"}
    for i in range(2, 11):
        paths[i] = root / "labels" / f"positioning_labels_fold_{i}.csv"

    missing = [i for i, p in paths.items() if not p.is_file()]
    if missing:
        print(f"ERROR: missing files for folds {missing}", file=sys.stderr)
        return 1

    splits: dict[int, dict[str, set[str]]] = {}
    for i, p in paths.items():
        df = pd.read_csv(p)
        splits[i] = sop_sets_by_split(df)

    print("=== 1) Test sets (SOP level) ===")
    all_ok = True
    if args.cv_style == "rotating_9":
        for a, b in combinations(range(1, 11), 2):
            overlap = splits[a]["Test"] & splits[b]["Test"]
            if overlap:
                print(f"  FAIL: Fold {a} & Fold {b} test: {len(overlap)} SOPs (sample): {list(overlap)[:3]}")
                all_ok = False
        print("  OK: all 45 pairs disjoint" if all_ok else "  FAIL: test overlap")
    else:
        t0 = splits[1]["Test"]
        for i in range(2, 11):
            if splits[i]["Test"] != t0:
                print(f"  FAIL: fold {i} Test SOP set differs from fold 1 (|symdiff|={len(splits[i]['Test'] ^ t0)})")
                all_ok = False
        print("  OK: all folds share identical Test SOP set" if all_ok else "  FAIL: test set mismatch")

    print("\n=== 2) Per-fold universe & Test coverage ===")
    per_fold_universe: dict[int, set[str]] = {}
    for i in range(1, 11):
        u = splits[i]["Train"] | splits[i]["Validation"] | splits[i]["Test"]
        per_fold_universe[i] = u
        if len(u) != 2000:
            print(f"  WARN: fold {i} Train∪Val∪Test has {len(u)} SOPs (expected 2000)")

    u0 = per_fold_universe[1]
    same_universe = all(per_fold_universe[i] == u0 for i in range(2, 11))
    print(f"  All folds same 2000-SOP universe: {same_universe}")

    union_test = set().union(*[splits[i]["Test"] for i in range(1, 11)])
    print(f"  Unique SOPs in dataset (fold 1): {len(u0)}")
    print(f"  SOPs that appear as Test in at least one fold file: {len(union_test)}")
    never_tested = u0 - union_test
    print(f"  SOPs never marked Test in any fold CSV: {len(never_tested)}")
    if never_tested:
        print(f"    examples: {list(never_tested)[:5]}")
    if args.cv_style == "rotating_9" and all_ok and len(union_test) == len(u0) and len(never_tested) == 0:
        print("  OK: tests partition the 2000 SOPs (each SOP is Test in exactly one fold CSV)")
    elif args.cv_style == "fixed_test" and all_ok:
        print(f"  OK: fixed test — union size {len(union_test)} (same 200 SOPs repeated across folds)")
    elif args.cv_style == "rotating_9":
        print("  WARN: coverage/partition does not match rotating_9 expectations")

    coverage_ok = (args.cv_style == "fixed_test" and len(never_tested) == 0) or (
        args.cv_style == "rotating_9" and len(union_test) == len(u0) and len(never_tested) == 0
    )

    print("\n=== 3) Validation SOP sets across folds ===")
    val_sets = [splits[i]["Validation"] for i in range(1, 11)]
    all_same = all(v == val_sets[0] for v in val_sets[1:])
    print(f"  All Validation SOP sets identical: {all_same}")
    print(f"  |Val| per fold: {[len(v) for v in val_sets]}")
    if not all_same:
        for i in range(2, 11):
            inter = len(val_sets[0] & val_sets[i - 1])
            print(f"  |Val fold1 & Val fold{i}|: {inter}")
    print(
        "  Note: Good/Bad *counts* in Val can match every fold while SOP membership "
        "differs (stratified resample with random_state=fold in cv_splits rotating_9)."
    )

    print("\n=== 4) Within-fold leakage (Train / Val / Test disjoint) ===")
    leak_any = False
    for i in range(1, 11):
        tr, va, te = splits[i]["Train"], splits[i]["Validation"], splits[i]["Test"]
        a, b, c = len(tr & te), len(tr & va), len(va & te)
        if a or b or c:
            print(f"  FAIL fold {i}: train&test={a}, train&val={b}, val&test={c}")
            leak_any = True
    if not leak_any:
        print("  OK: no within-fold overlap")

    print("\n=== 5) Row-level Validation Good/Bad (sanity) ===")
    for i in (1, 2, 3, 10):
        df = pd.read_csv(paths[i])
        v = df[df["Split"] == "Validation"]
        g = (v["qualitativeLabel"] == "Good").sum()
        b = (v["qualitativeLabel"] == "Bad").sum()
        print(f"  fold {i}: Val rows={len(v)}, Good={g}, Bad={b}")

    return 0 if all_ok and not leak_any and coverage_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
