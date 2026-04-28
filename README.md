
# MLO Breast Positioning Assessment via Deep Learning

Breast cancer is a leading cause of cancer-related mortality in women worldwide, making early detection through mammography screening critically important. The effectiveness of mammography, however, depends heavily on accurate breast positioning. Suboptimal positioning can result in missed findings, increased patient discomfort, and unnecessary repeat imaging.

This repository implements a deep learning pipeline for quantitative assessment of mediolateral oblique (MLO) mammogram positioning quality. The pipeline detects anatomical landmarks — the nipple and pectoralis muscle — and automatically delineates the Posterior Nipple Line (PNL) to evaluate positioning quality. Two segmentation-based regression models (UNet and Attention UNet) and one classification model (ResNeXt50) are included.

## Repository Structure

```
├── code/
│   ├── classification/          # ResNeXt50 binary quality classifier
│   │   ├── main.py              # Training entry point
│   │   ├── test.py              # Inference + GradCAM visualization
│   │   └── utils/               # Dataloader, model, loss, metrics
│   └── regression/              # Landmark regression models (UNet / Attention UNet)
│       ├── main/
│       │   ├── main.py          # Training entry point
│       │   ├── evaluation.py    # Evaluation with distance metrics
│       │   ├── visualize_test_predictions.py
│       │   └── utils/           # Dataloader, models, loss, train, validate
│       └── preprocessing/       # DICOM preprocessing pipeline
└── labels/                      # Dataset annotation details
```

## Dataset

Labels were created on 1,000 randomly selected MLO mammograms from the [VinDr-Mammo](https://vindr.ai/datasets/mammo) open-access dataset. Annotations were performed by two board-certified breast radiologists with over five years of breast imaging experience.

| Split      | Automated PNL-based Quality | Expert Qualitative Label |
|:----------:|:---------------------------:|:------------------------:|
| Training   | 967 good, 633 poor          | 1,185 good, 415 poor     |
| Validation | 108 good, 92 poor           | 132 good, 68 poor        |
| Testing    | 123 good, 77 poor           | 146 good, 54 poor        |

See [`labels/README.md`](labels/README.md) for full annotation details.

## Models

### Regression Models (Landmark Detection → Quality Assessment)

Both models take grayscale mammographic images as input and predict the coordinates of the nipple and pectoralis muscle landmarks. Positioning quality (good/poor) is then derived automatically from the resulting PNL geometry.

- **UNet** — Encoder–decoder architecture with skip connections, adapted for landmark coordinate regression.
- **Attention UNet (RAUNet)** — Extends UNet with attention gates at each decoder level to focus on anatomically relevant spatial regions.

### Classification Model

- **ResNeXt50** — ResNeXt-50 (32×4d) fine-tuned for direct binary positioning quality classification (good / poor), adapted to accept single-channel (grayscale) input.

## Installation

```bash
git clone https://github.com/<your-username>/mlo-breast-positioning-assessment.git
cd mlo-breast-positioning-assessment
pip install torch torchvision pandas numpy scikit-learn matplotlib Pillow pydicom
```

## Usage

### Training — Regression Models

```bash
cd code/regression/main
python main.py --config configs/example_config.json
```

Set `model_type` to `"UNet"` or `"RAUNet"` in the config file.

### Training — Classification Model

```bash
cd code/classification
python main.py
```

Update the `config` dictionary in `main.py` with your dataset paths and hyperparameters.

### Evaluation — Regression

```bash
cd code/regression/main
python evaluation.py --config configs/example_eval_config.json
```

### Inference + GradCAM — Classification

```bash
cd code/classification
python test.py
```

GradCAM visualizations are saved to `gradcam_outs/`.

## Hyperparameters

| Hyperparameter | Regression Models (UNet / Attention UNet)        | Classification Model (ResNeXt50) |
|----------------|--------------------------------------------------|----------------------------------|
| Task           | Landmark coordinate regression (Pec1, Pec2, Nipple) | Binary quality classification (Good / Poor) |
| Optimizer      | AdamW                                            | Adam                             |
| Batch Size     | 32                                               | 32                               |
| Epochs         | 300                                              | 30                               |
| Learning Rate  | CyclicLR (base: 1e-5, max: 5e-4)                | Fixed: 1e-4                      |
| Loss Function  | Weighted combination (MSE + MAE + SmoothL1 + Wing) | Cross-Entropy Loss             |

## Performance

### Landmark Distance Errors (mm)

Distance errors are reported as mean (μ), standard deviation (σ), and median (x̃) in millimeters on the test set.

| Model          | Perp μ | Perp σ | Perp x̃ | Pec1 μ | Pec1 σ | Pec1 x̃ | Pec2 μ | Pec2 σ | Pec2 x̃ | Nipple μ | Nipple σ | Nipple x̃ | Angular μ | Angular σ | Angular x̃ |
|----------------|--------|--------|---------|--------|--------|---------|--------|--------|---------|----------|----------|-----------|-----------|-----------|------------|
| UNet           | 9.79   | 6.57   | 8.63    | 8.43   | 7.16   | 6.68    | 13.34  | 10.76  | 10.62   | 5.57     | 4.36     | 4.62      | 3.81      | 3.11      | 2.93       |
| Attention UNet | **6.00** | **5.41** | **4.45** | **6.88** | **6.17** | **5.12** | **8.85** | **9.89** | **6.50** | **3.01** | **2.53** | **2.33** | **2.94** | **2.61** | **2.26** |

### Landmark Pixel Errors

| Model          | Perp (Mean ± Std) | Pec1 (Mean ± Std) | Pec2 (Mean ± Std) | Nipple (Mean ± Std) | Angular (°) Mean ± Std | Angular (°) Median |
|----------------|-------------------|-------------------|-------------------|---------------------|------------------------|--------------------|
| UNet           | 20.39 ± 14.26     | 17.16 ± 14.65     | 27.55 ± 22.47     | 11.56 ± 9.37        | 3.81 ± 3.11            | 2.93               |
| Attention UNet | **12.44 ± 11.33** | **14.02 ± 12.51** | **18.37 ± 20.67** | **6.32 ± 5.50**     | **2.94 ± 2.61**        | **2.26**           |

### Positioning Quality Classification

Positioning quality (good / poor) is derived from the predicted landmarks via the PNL rule for the regression models, and predicted directly for ResNeXt50.

| Model          | Accuracy (%) | Sensitivity (%) | Specificity (%) |
|----------------|--------------|-----------------|-----------------|
| ResNeXt50      | 72.00        | 37.04           | 84.93           |
| UNet           | 82.00        | 71.43           | 88.62           |
| Attention UNet | **85.50**    | **84.42**       | **86.18**       |

## Example Predictions

### Model Comparison on Test Sample

The figure below shows predictions from all three models on the same test mammogram. The Attention UNet accurately localizes the nipple and pectoralis muscle landmarks, from which the PNL is automatically derived.

<p align="center">
  <img width="950" alt="Model comparison on test sample 198" src="assets/combined_models_198.png">
</p>

### Pipeline Overview — Good vs. Poor Positioning

The top row shows well-positioned mammograms: the predicted PNL aligns with the pectoralis muscle line, and the classification model correctly identifies them as good. The bottom row shows poorly positioned cases: PNL and pectoralis line diverge, and the classification model correctly flags them as poor.

<p align="center">
  <img width="950" alt="Pipeline overview: good vs poor positioning examples" src="assets/Figure3.png">
</p>


## Contributing

Contributions are welcome. For significant changes, please open an issue first to discuss the proposed modification.
