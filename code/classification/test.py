import argparse
import json
import os

import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

from utils.dataloader import get_dataloader
from utils.metrics import gradcam_visualization, evaluate_model


DEFAULT_CONFIG = {
    'image_dir': '../regression/data/images',
    'annotations_file': '../../labels/positioning_labels.csv',
    'batch_size': 8,
    'model_name': 'resnext50',
    'num_classes': 2,
    'label_type': 'manual',  # 'manual' or 'automated'
    'best_model_path': 'models/best_model.pth',
    'gradcam_output_dir': 'gradcam_outs',
    'metrics_csv': 'test_metrics.csv',
    'results_csv': 'predictions.csv',
    'eval_json_path': None,  # set in per-fold configs to also emit JSON consumed by run_cv_statistics.py
}


def load_config(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def write_eval_json(results_csv, json_path):
    """Convert the per-image predictions CSV into the JSON format expected by
    statistical_test/run_cv_statistics.py (per_image list keyed by UIDs)."""
    preds_df = pd.read_csv(results_csv)
    per_image = [
        {
            'StudyInstanceUID': str(row['StudyInstanceUID']),
            'SOPInstanceUID': str(row['SOPInstanceUID']),
            'prediction_label': int(row['Prediction']),
        }
        for _, row in preds_df.iterrows()
    ]
    os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'per_image': per_image}, f, indent=2)
    print(f"Per-image predictions JSON saved to {json_path}")


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = get_dataloader(
        config['image_dir'], config['annotations_file'], 'Test',
        config['label_type'], config['batch_size'],
    )

    # ResNeXt50 with single-channel input head, num_classes output
    model = models.resnext50_32x4d(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
    model = model.to(device)
    model.load_state_dict(torch.load(config['best_model_path'], map_location=device))

    # GradCAM is only meaningful for the baseline qualitative inspection; per-fold
    # runs (driven by configs/ResNeXt50_fold_{k}_eval.json) leave it off.
    if config.get('gradcam_output_dir'):
        gradcam_visualization(model, test_loader, device, output_dir=config['gradcam_output_dir'])

    metrics_csv = config.get('metrics_csv', 'test_metrics.csv')
    results_csv = config.get('results_csv', 'predictions.csv')
    for p in (metrics_csv, results_csv):
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)
    evaluate_model(
        model, test_loader, device, config['num_classes'],
        metrics_csv=metrics_csv,
        results_csv=results_csv,
        annotation_path=config['annotations_file'],
    )

    eval_json_path = config.get('eval_json_path')
    if eval_json_path:
        write_eval_json(results_csv, eval_json_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ResNeXt50 positioning quality classifier — inference + GradCAM")
    parser.add_argument('--config', type=str, default=None,
                        help="Optional path to per-fold JSON config (e.g. configs/ResNeXt50_fold_2_eval.json). "
                             "If omitted, runs the baseline test (with GradCAM).")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        # Per-fold configs do not set GradCAM output by default; skip it.
        cfg.setdefault('gradcam_output_dir', None)
        # Derive a sibling predictions CSV next to the eval JSON so the JSON
        # writer can stream rows back through it.
        if cfg.get('eval_json_path'):
            base = os.path.splitext(cfg['eval_json_path'])[0]
            cfg.setdefault('results_csv', f"{base}_predictions.csv")
            cfg.setdefault('metrics_csv', f"{base}_metrics.csv")
    else:
        cfg = dict(DEFAULT_CONFIG)
    main(cfg)
