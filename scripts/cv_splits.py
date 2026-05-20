#!/usr/bin/env python3
"""
Generate positioning_labels_fold_{k}.csv for k >= 2.

Modes:
  fixed_test   — Test rows identical to positioning_labels.csv on every fold;
                 only Train/Validation reshuffled.
  rotating_9   — Fold 1 = original CSV (unchanged; the original held-out test
                 partition is exam-aligned, 200 images from 100 examinations).
                 Folds 2–10 use nine disjoint test sets (200 SOPs each) drawn
                 from the 1,800 non-test SOPs of the original file, stratified
                 on the qualitative label. Train+Val for those folds = the
                 remaining 1,800; the Validation set preserves the original
                 Good/Bad counts and is resampled per fold (random_state=fold).

K-fold property: the union of the ten test partitions covers the 2,000-SOP
pool exactly once and the partitions are pairwise disjoint at the
SOPInstanceUID level. Folds 2–10 partition at the image (SOPInstanceUID)
level — consistent with the per-image positioning-quality label — so the
two MLO views of an examination may fall in different folds for those
nine folds. See the manuscript §2 Statistical Analyses and §4 Discussion
for the methodological rationale.

Does not modify labels/positioning_labels.csv.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_uid_pairs(df: pd.DataFrame) -> set[tuple[str, str]]:
    t = df[(df["Split"] == "Test") & (df["labelName"] == "Pectoralis")]
    return set(zip(t["StudyInstanceUID"].astype(str), t["SOPInstanceUID"].astype(str)))


def pectoralis_meta(pool_df: pd.DataFrame) -> pd.DataFrame:
    return pool_df[pool_df["labelName"] == "Pectoralis"][
        ["SOPInstanceUID", "StudyInstanceUID", "qualitativeLabel", "Split"]
    ].drop_duplicates(subset=["SOPInstanceUID"])


def stratified_val_pick(
    pectoralis: pd.DataFrame,
    n_val_good: int,
    n_val_bad: int,
    rng: np.random.RandomState,
) -> tuple[set[str], set[str]]:
    val_sops: set[str] = set()
    for label, n_take in [("Good", n_val_good), ("Bad", n_val_bad)]:
        sops_label = pectoralis.loc[pectoralis["qualitativeLabel"] == label, "SOPInstanceUID"].astype(str).values
        if len(sops_label) < n_take:
            raise RuntimeError(f"Not enough {label} in pool for validation (need {n_take}, have {len(sops_label)}).")
        perm = rng.permutation(len(sops_label))
        val_sops.update(sops_label[perm[:n_take]])
    all_sops = set(pectoralis["SOPInstanceUID"].astype(str))
    train_sops = all_sops - val_sops
    if len(val_sops) != n_val_good + n_val_bad or (train_sops & val_sops):
        raise RuntimeError("Train/val SOP assignment invalid.")
    return train_sops, val_sops


def build_fold_csv_fixed_test(original: pd.DataFrame, fold: int) -> pd.DataFrame:
    if fold < 2:
        raise ValueError("fold must be >= 2 (fold 1 is the original CSV).")
    seed = fold
    pool = original[original["Split"] != "Test"].copy()
    pectoralis = pectoralis_meta(pool)
    val_mask = pectoralis["Split"] == "Validation"
    n_val_good = int((pectoralis.loc[val_mask, "qualitativeLabel"] == "Good").sum())
    n_val_bad = int((pectoralis.loc[val_mask, "qualitativeLabel"] == "Bad").sum())
    rng = np.random.RandomState(seed)
    train_sops, val_sops = stratified_val_pick(pectoralis, n_val_good, n_val_bad, rng)

    def new_split(row: pd.Series) -> str:
        if row["Split"] == "Test":
            return "Test"
        sop = str(row["SOPInstanceUID"])
        if sop in val_sops:
            return "Validation"
        return "Train"

    out = original.copy()
    out["Split"] = out.apply(new_split, axis=1)
    return out


def nine_disjoint_test_sets_from_pool(
    pool_pectoralis: pd.DataFrame,
    base_seed: int,
) -> list[set[str]]:
    """
    Partition the 1800 non-test SOPs into 9 disjoint test sets of 200 SOPs each.

    Stratified: per-fold Good/Bad counts are either (146, 54) or (147, 53) so that
    totals match the pool (1317 Good, 483 Bad) and every test fold sums to 200.
    This matches the original fold-1 test class mix (146 Good, 54 Bad) up to ±1.
    """
    rng = np.random.RandomState(base_seed)
    goods = pool_pectoralis.loc[
        pool_pectoralis["qualitativeLabel"] == "Good", "SOPInstanceUID"
    ].astype(str).tolist()
    bads = pool_pectoralis.loc[
        pool_pectoralis["qualitativeLabel"] == "Bad", "SOPInstanceUID"
    ].astype(str).tolist()
    rng.shuffle(goods)
    rng.shuffle(bads)
    g_total, b_total = len(goods), len(bads)
    if g_total + b_total != 1800:
        raise RuntimeError(f"rotating_9 expects 1800 non-test SOPs, got {g_total + b_total}")

    b0, r_extra = divmod(b_total, 9)
    # r_extra folds get (b0+1) Bad and (200 - b0 - 1) Good; the rest get b0 Bad and (200 - b0) Good.
    n_bad_high = int(r_extra)
    n_bad_low = 9 - n_bad_high
    bad_counts = [b0 + 1] * n_bad_high + [b0] * n_bad_low
    rng.shuffle(bad_counts)
    good_counts = [200 - b for b in bad_counts]
    if sum(good_counts) != g_total or sum(bad_counts) != b_total:
        raise RuntimeError("Internal error: stratified 9-fold counts do not match pool.")

    out: list[set[str]] = []
    gi = bi = 0
    for ng, nb in zip(good_counts, bad_counts):
        chunk_g = goods[gi : gi + ng]
        chunk_b = bads[bi : bi + nb]
        gi += ng
        bi += nb
        out.append(set(chunk_g) | set(chunk_b))
    if gi != g_total or bi != b_total:
        raise RuntimeError("Chunking did not consume full Good/Bad pools.")
    if any(len(s) != 200 for s in out):
        raise RuntimeError(f"Expected 9 x 200 test SOPs, got sizes {[len(s) for s in out]}")
    union = set().union(*out)
    if union != set(goods) | set(bads):
        raise RuntimeError("9-fold partition does not cover the 1800-SOP pool.")
    for i in range(9):
        for j in range(i + 1, 9):
            if out[i] & out[j]:
                raise RuntimeError("Partition overlap between test folds.")
    return out


def build_fold_csv_rotating9(original: pd.DataFrame, fold: int, partition_seed: int) -> pd.DataFrame:
    """
    fold in 2..10 only. Fold 1 is always the original file.
    Test for fold k = nine_partitions[k-2] (200 SOPs from original non-test pool).
    Train+val = remaining 1800 SOPs; val size/stratification matches original Train+Val split.
    """
    if fold < 2 or fold > 10:
        raise ValueError("rotating_9 mode only supports folds 2..10 (fold 1 = original CSV).")

    orig_test_mask = original["Split"] == "Test"
    pool_df = original[~orig_test_mask].copy()
    pool_p = pectoralis_meta(pool_df)
    nine_tests = nine_disjoint_test_sets_from_pool(pool_p, base_seed=partition_seed)
    test_sops = nine_tests[fold - 2]

    train_val_mask = ~original["SOPInstanceUID"].astype(str).isin(test_sops)
    train_val_pectoralis = pectoralis_meta(original[train_val_mask])

    val_ref = pectoralis_meta(original[original["Split"] == "Validation"])
    n_val_good = int((val_ref["qualitativeLabel"] == "Good").sum())
    n_val_bad = int((val_ref["qualitativeLabel"] == "Bad").sum())

    rng = np.random.RandomState(fold)
    train_sops, val_sops = stratified_val_pick(train_val_pectoralis, n_val_good, n_val_bad, rng)

    def new_split(row: pd.Series) -> str:
        sop = str(row["SOPInstanceUID"])
        if sop in test_sops:
            return "Test"
        if sop in val_sops:
            return "Validation"
        if sop in train_sops:
            return "Train"
        raise KeyError(f"SOP {sop} unassigned")

    out = original.copy()
    out["Split"] = out.apply(new_split, axis=1)
    return out


def print_stats(df: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")
    for split in ["Train", "Validation", "Test"]:
        sub = df[df["Split"] == split]
        n = len(sub)
        good = (sub["qualitativeLabel"] == "Good").sum()
        bad = (sub["qualitativeLabel"] == "Bad").sum()
        g_pct = 100.0 * good / n if n else 0.0
        b_pct = 100.0 * bad / n if n else 0.0
        print(f"  {split}: rows={n}, Good={good} ({g_pct:.2f}%), Bad={bad} ({b_pct:.2f}%)")


def parse_folds(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="CV split CSV generator")
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to positioning_labels.csv (default: <repo>/labels/positioning_labels.csv)",
    )
    parser.add_argument(
        "--mode",
        choices=("fixed_test", "rotating_9"),
        default="rotating_9",
        help="fixed_test: same Test as original on every fold. rotating_9 (default): fold 1=file copy; "
        "folds 2–10 use 9 disjoint tests from the original 1800 non-test SOPs.",
    )
    parser.add_argument(
        "--partition-seed",
        type=int,
        default=0,
        help="For rotating_9: RNG seed when building the 9 test blocks from the 1800-SOP pool (default: 0).",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="2-10",
        help='Fold indices to write, e.g. "2-10" or "3,5,7"',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats in memory; no file writes",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write labels/positioning_labels_fold_{k}.csv for selected folds",
    )
    args = parser.parse_args()

    root = repo_root()
    labels_path = args.labels or (root / "labels" / "positioning_labels.csv")
    if not labels_path.is_file():
        print(f"ERROR: labels file not found: {labels_path}", file=sys.stderr)
        return 1

    original = pd.read_csv(labels_path)

    def build(fold: int) -> pd.DataFrame:
        if args.mode == "fixed_test":
            return build_fold_csv_fixed_test(original, fold)
        return build_fold_csv_rotating9(original, fold, partition_seed=args.partition_seed)

    if args.dry_run and not args.write:
        print_stats(original, "Fold 1 (original CSV) row-level qualitativeLabel")
        f2 = build(2)
        f3 = build(3)
        print_stats(f2, f"Fold 2 simulated ({args.mode})")
        print_stats(f3, f"Fold 3 simulated ({args.mode})")
        if args.mode == "fixed_test":
            orig_test = test_uid_pairs(original)
            if test_uid_pairs(f2) != orig_test:
                print("FAIL: fixed_test fold2 test != original.", file=sys.stderr)
                return 2
            print("\nOK (fixed_test): Fold 2 Test UIDs match original.")
        else:
            t2 = set(f2[f2["Split"] == "Test"]["SOPInstanceUID"].astype(str))
            t3 = set(f3[f3["Split"] == "Test"]["SOPInstanceUID"].astype(str))
            t1 = set(original[original["Split"] == "Test"]["SOPInstanceUID"].astype(str))
            print("\n=== Rotating test checks (SOP level) ===")
            print(f"  |Test fold2|: {len(t2)}, |Test fold3|: {len(t3)}, |Test orig|: {len(t1)}")
            print(f"  fold2 & fold3: {len(t2 & t3)} (expect 0)")
            print(f"  fold2 & orig:  {len(t2 & t1)} (expect 0)")
            if t2 & t3 or t2 & t1 or t3 & t1:
                print("FAIL: unexpected test overlap.", file=sys.stderr)
                return 2
            print("  OK: disjoint tests across fold2, fold3, and original.")
            print("\n=== Full pairwise Test disjoint (folds 1–10) ===")
            test_sets: dict[int, set[str]] = {
                1: set(original[original["Split"] == "Test"]["SOPInstanceUID"].astype(str))
            }
            for fd in range(2, 11):
                test_sets[fd] = set(build(fd)[build(fd)["Split"] == "Test"]["SOPInstanceUID"].astype(str))
            full_ok = True
            for a, b in combinations(range(1, 11), 2):
                if test_sets[a] & test_sets[b]:
                    print(f"  FAIL: fold {a} & fold {b} test = {len(test_sets[a] & test_sets[b])}")
                    full_ok = False
            print("  OK: all 45 pairs disjoint" if full_ok else "  FAIL: see above")
            ratios = []
            for fd in range(2, 11):
                dfx = build(fd)
                ts = dfx[dfx["Split"] == "Test"]
                g = (ts["qualitativeLabel"] == "Good").sum()
                b = (ts["qualitativeLabel"] == "Bad").sum()
                ratios.append((fd, g, b, 100.0 * g / len(ts) if len(ts) else 0))
            print("\n=== Test split (rows) folds 2–10: Good / Bad / Good% ===")
            for fd, g, b, gp in ratios:
                print(f"  fold {fd}: Good={g}, Bad={b}, Good%={gp:.2f}")
            g_pct_set = {r[3] for r in ratios}
            print(f"  Unique Good% values: {sorted(g_pct_set)}")
        print("\nDry-run complete (no files written).")
        return 0

    folds = parse_folds(args.folds)

    if not args.write:
        print("Nothing to do: pass --write to save CSVs, or --dry-run for in-memory report.")
        return 0

    print_stats(original, "Reference: original CSV (fold 1)")
    for fold in folds:
        if fold < 2:
            print(f"Skip fold {fold} (< 2).", file=sys.stderr)
            continue
        if args.mode == "rotating_9" and fold > 10:
            print(f"Skip fold {fold} (> 10 in rotating_9).", file=sys.stderr)
            continue
        out_df = build(fold)
        out_path = root / "labels" / f"positioning_labels_fold_{fold}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(out_df)} rows)")
        print_stats(out_df, f"Written fold {fold} ({args.mode})")
        if args.mode == "fixed_test" and fold == 2:
            if test_uid_pairs(out_df) != test_uid_pairs(original):
                print("ERROR: fixed_test fold 2 Test UIDs do not match original.", file=sys.stderr)
                return 2
            print("  OK: Test UIDs match original (fixed_test).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
