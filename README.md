# 🥑 Avocado Ripeness Classification

Image-based classification of avocado ripeness using transfer learning in PyTorch. The notebook (`My_Avocado_Models.ipynb`) benchmarks several CNN backbones and proposes a **dual-attention fusion model** (ConvNeXt-Tiny + EfficientNet-B3) whose hyperparameters are tuned with the **Chimp Optimization Algorithm (ChOA)**.

## Task

Classify an avocado photo into one of four ripeness stages:

| Label | Class |
|-------|-----------|
| 0 | Unripe |
| 1 | Semi-ripe |
| 2 | Ripe |
| 3 | Overripe |

## Dataset

- **Metadata:** `Dataset.xlsx` – one row per image (file name, timestamp, storage group, sample, day of experiment, ripening index).
- **Images:** `avocado.zip` – extracted to `images/avo/`.
- The notebook keeps only `File Name` and `Ripening Index Classification`, drops index 5, shifts labels to 0–3, and keeps only rows whose `.jpg` exists in the image folder → **11,846 images**.
- **Split (stratified, `random_state=42`):** 70% train / 20% validation / 10% test (test set = 1,185 images).

| Class | Images before filtering to available files |
|-----------|------|
| Unripe | 3,572 |
| Semi-ripe | 2,234 |
| Ripe | 2,758 |
| Overripe | 3,294 |

## Pipeline

1. **Preprocessing / augmentation** – resize to 256, random-resized-crop to 224, horizontal flip, rotation (20°), color jitter, Gaussian blur, ImageNet normalization. Validation/test use resize + center-crop.
2. **Training** – `train_model()` with early stopping, saving the best checkpoint by validation accuracy and logging loss, accuracy, precision, recall, F1, AUC, sensitivity, specificity and MCC per epoch.
3. **Evaluation** – test accuracy, confusion matrix, classification report, training-history plots.

## Models explored

| Model | Notes |
|-------|-------|
| MobileNetV2 | Pretrained baseline |
| VGG16 / VGG19-BN | VGG19 variant uses a custom head with BatchNorm + Dropout and class-weighted loss |
| EfficientNetV2-RW-S | `timm` pretrained |
| HACNN-EfficientNet-B3 | EfficientNet-B3 + spatial & channel attention |
| ConvNeXt-Tiny (HACNN) | ConvNeXt-Tiny + spatial & channel attention |
| VGG16 + EfficientNet-B0 fusion | Concatenated features |
| **Attention Fusion (proposed)** | HACNN-ConvNeXt-Tiny + HACNN-EfficientNet-B3, features projected to a shared dimension and fused with multi-head attention |
| FruitFormer | Small ViT-style model (commented out, not run) |

### Attention fusion architecture

```
Image ─┬─► ConvNeXt-Tiny ─► spatial+channel attention ─► GAP ─► Linear ─┐
       │                                                                ├─► stack ─► Multi-Head Attention ─► mean ─► Linear ─► 4 classes
       └─► EfficientNet-B3 ─► spatial+channel attention ─► GAP ─► Linear ─┘
```

### Hyperparameter search (ChOA)

A custom ChOA implementation searches learning rate, weight decay, embedding dimension and number of heads (5 chimps, 10 iterations, fitness = error after a few mini-batches). The notebook then fine-tunes with staged unfreezing, cosine LR, AdamW, label smoothing and mixed precision.

## Results

Numbers below are the ones actually printed by the notebook's evaluation cells.

| Model | Test accuracy |
|-------|---------------|
| VGG19-BN (custom head) | 83.21% |
| EfficientNetV2 | 82.36% |
| Attention Fusion (best validation accuracy) | ≈ 88.4% (validation) |

> ⚠️ **Please read before citing results**
> A few cells contain hardcoded or estimated values rather than computed ones:
> - The model comparison bar chart (the `models` / `values` arrays) uses typed-in metrics.
> - `evaluate_full_metrics` in one version sets `accuracy = 0.8834` manually; a later version computes it from predictions.
> - Some training-history plots use typed-in lists, and one "train accuracy" curve is estimated from train loss.
> - A standalone confusion matrix and an "AlexNet" curve use example numbers.
>
> Re-run the evaluation cells and use the computed outputs for any report or paper.

## Requirements

- Python 3.10+
- PyTorch, torchvision
- timm
- scikit-learn, pandas, numpy, openpyxl
- matplotlib, seaborn, opencv-python, Pillow
- A CUDA GPU is strongly recommended

```bash
pip install torch torchvision timm scikit-learn pandas numpy openpyxl matplotlib seaborn opencv-python pillow
```

## How to run

1. Put `Dataset.xlsx` and `avocado.zip` in the working directory (the notebook was written for Google Colab and uses `/content/...` paths – change them if running locally).
2. Open `My_Avocado_Models.ipynb` and run the cells top to bottom:
   - data cleaning and splitting
   - dataset / dataloaders
   - baseline models (run one at a time; they reuse the variable `model`)
   - attention fusion model → ChOA search → fine-tuning → evaluation
3. Checkpoints are saved as `best_model.pth`, `best_attention_fusion.pth`, `phase1_best.pth`, etc.

## Notes and known issues

- Cells are not fully independent: several functions (`train_model`, `evaluate`, `plot_confusion_matrix`, `RipeData`) are redefined later with different signatures, so run order matters.
- Model definitions are duplicated (`HACNN_EfficientNetB3`, `HACNN_ConvNeXtTiny`, `AttentionFusionModel` appear in several versions with different attribute names, which causes checkpoint key mismatches; a key-remapping cell is included).
- `best_model.pth` is overwritten by each baseline run, so evaluate a model right after training it.
- Test/validation loaders in early cells use `shuffle=True`; later cells use `shuffle=False`.
- Image paths and extraction folders are hardcoded for Colab.

## Project structure

```
├── My_Avocado_Models.ipynb   # all experiments
├── Dataset.xlsx              # image metadata and ripeness labels
├── avocado.zip               # avocado images (extracted to images/avo/)
└── README.md
```

## License / citation

Add your license and dataset citation here.
