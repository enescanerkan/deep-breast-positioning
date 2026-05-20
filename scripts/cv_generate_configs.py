#!/usr/bin/env python3
"""Generate fold-specific training and eval JSON configs under <repo>/configs/."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_folds(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=str, default="2-10", help='e.g. "2-10" or "2,3,4"')
    args = parser.parse_args()
    folds = parse_folds(args.folds)

    reg_example = root / "code" / "regression" / "main" / "configs" / "example_config.json"
    reg_eval_example = root / "code" / "regression" / "main" / "configs" / "example_eval_config.json"
    if not reg_example.is_file() or not reg_eval_example.is_file():
        print("Missing regression example configs.", file=sys.stderr)
        return 1

    base_train = json.loads(reg_example.read_text(encoding="utf-8"))
    base_eval = json.loads(reg_eval_example.read_text(encoding="utf-8"))

    out_dir = root / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        if fold < 2:
            continue
        split_rel = f"../../../labels/positioning_labels_fold_{fold}.csv"
        for model in ("UNet", "RAUNet"):
            cfg = deepcopy(base_train)
            cfg["model_type"] = model
            cfg["split_file"] = split_rel
            cfg["best_model_path"] = f"../../../models/{model}_fold_{fold}.pth"
            (out_dir / f"{model}_fold_{fold}.json").write_text(
                json.dumps(cfg, indent=4) + "\n", encoding="utf-8"
            )

            ev = deepcopy(base_eval)
            ev["model_type"] = model
            ev["split_file"] = split_rel
            ev["best_model_path"] = f"../../../models/{model}_fold_{fold}.pth"
            ev["eval_json_path"] = f"../../../results/{model}_fold_{fold}_eval.json"
            ev["eval_artifact_stem"] = f"../../../results/{model}_fold_{fold}_eval"
            (out_dir / f"{model}_fold_{fold}_eval.json").write_text(
                json.dumps(ev, indent=4) + "\n", encoding="utf-8"
            )

        cls_cfg = {
            "model_name": "resnext50",
            "label_type": "manual",
            "num_classes": 2,
            "batch_size": 8,
            "num_epochs": 30,
            "patience": 10,
            "base_lr": 1e-5,
            "max_lr": 5e-4,
            "step_size_down": 10,
            "learning_rate": 1e-4,
            "device": "cuda",
            "pretrained": True,
            "image_dir": "../regression/data/images",
            "annotations_file": f"../../labels/positioning_labels_fold_{fold}.csv",
            "best_model_path": f"../../models/ResNeXt50_fold_{fold}.pth",
            "eval_json_path": f"../../results/ResNeXt50_fold_{fold}_eval.json",
        }
        (out_dir / f"ResNeXt50_fold_{fold}.json").write_text(
            json.dumps(cls_cfg, indent=4) + "\n", encoding="utf-8"
        )
        cls_eval = {
            "model_name": "resnext50",
            "label_type": "manual",
            "num_classes": 2,
            "batch_size": 8,
            "device": "cuda",
            "image_dir": "../regression/data/images",
            "annotations_file": f"../../labels/positioning_labels_fold_{fold}.csv",
            "best_model_path": f"../../models/ResNeXt50_fold_{fold}.pth",
            "eval_json_path": f"../../results/ResNeXt50_fold_{fold}_eval.json",
        }
        (out_dir / f"ResNeXt50_fold_{fold}_eval.json").write_text(
            json.dumps(cls_eval, indent=4) + "\n", encoding="utf-8"
        )

    print(f"Wrote configs for folds {folds} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
