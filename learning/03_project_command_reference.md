# Lesson 03 — Definitive Project Command Reference Guide

> **Role**: Explain Agent (ML Systems Engineer + Teacher)  
> **Topic**: CLI Reference, Shell Utilities, PyTorch Execution Pipelines, and Git Workflows for Dental Classifier Study  
> **Target Audience**: Researcher / Paper Writing Reference Folder

This guide provides a central, single source of truth for **every command** used in this project's lifecycle, from dataset deduplication and augmentation to multi-model training, standalone evaluation, and paper report compilation.

---

## 1. Google Colab & Google Drive Setup

These commands are run inside your Google Colab notebook (`intraview-classifier.ipynb`) to initialize the cloud GPU runtime.

### 1.1 Mount Google Drive
Mounts your personal Google Drive account to persist datasets, training weights, and plots across runtime restarts.
```python
from google.colab import drive
drive.mount('/content/drive')
```
* **Why**: Google Colab workspaces are transient (deleted on timeout). Mounting Drive guarantees all output weights and logs are permanently saved to the cloud.

### 1.2 Repository Syncing
Clones the codebase to Colab or pulls the latest updates from your local workspace pushed to GitHub.
```bash
# Clone the repository (run once on first run)
!git clone https://github.com/AtharvaKuldhar/intraview-classifier.git
%cd intraview-classifier

# Pull latest local code changes pushed from Antigravity IDE
!git pull
```

### 1.3 Map Dataset Symlink (Crucial)
Creates a local symbolic link (`data/augmented`) mapping Colab's project workspace directory directly to your persistent Google Drive folder.
```bash
# Create local project directory
!mkdir -p data

# Symlink your Google Drive augmented data folder inside the project data directory
!ln -sf /content/drive/MyDrive/dental_research/datasets/data/augmented data/augmented

# Verify mapping (should list all 8 classes alphabetically)
!ls data/augmented
```
* **Why**: Avoids copying thousands of image files (~300MB) directly to Colab's workspace on every runtime boot, which takes valuable execution time. The symlink lets PyTorch load files on-demand directly from Drive.

### 1.4 Dependency Installation
```bash
!pip install timm albumentations pyyaml tensorboard pandas numpy matplotlib scikit-learn
```

---

## 2. Pre-Training Phase (Data Cleaning & Augmentation)

These commands were used locally (or can be run in Colab) to process the raw clinical images.

### 2.1 Deduplication & Cleaning
Crawl raw dataset folders, compute MD5 hashes, remove duplicate clinical views, resolve cross-class conflicts, and copy unique images to `data/processed/`.
```bash
python src/augmentations/deduplicate.py
```

### 2.2 Tier-Based Data Augmentation
Read deduplicated images from `data/processed/` and generate a perfectly balanced dataset of **5,234 images** using albumentations transforms inside `data/augmented/`.
```bash
python src/augmentations/augment.py
```

### 2.3 Augmentation Validation & Samples Grid
Verifies output image formats, counts class splits, checks for duplicate hashes, and plots a sample visual grid comparison.
```bash
python src/augmentations/verify_augmentation.py
```
* **Output Grid Location**: `outputs/plots/augmentation_samples.png`

---

## 3. Training Commands (Phase 1 & 2 Transfer Learning)

### 3.1 Single Model Training (CLI)
To train any of the 8 comparative model architectures individually, execute the CLI training runner pointing to the respective config file:

```bash
# 1. ResNet-50
!python src/training/train.py --config src/configs/resnet50.yaml

# 2. Swin Transformer Tiny
!python src/training/train.py --config src/configs/swin_tiny.yaml

# 3. DenseNet-121
!python src/training/train.py --config src/configs/densenet121.yaml

# 4. MobileNetV3-Small
!python src/training/train.py --config src/configs/mobilenetv3_small.yaml

# 5. EfficientNet-B2
!python src/training/train.py --config src/configs/efficientnet_b2.yaml

# 6. EfficientNet-B3
!python src/training/train.py --config src/configs/efficientnet_b3.yaml

# 7. ConvNeXt-Tiny
!python src/training/train.py --config src/configs/convnext_tiny.yaml

# 8. DINOv2-Small (Self-Supervised Vision Transformer)
!python src/training/train.py --config src/configs/dinov2_small.yaml
```

### 3.2 Dynamic Output Redirection Flag
Our script defaults to redirecting outputs to Drive on Colab. If you ever want to force it to write locally inside Colab workspace (or local machine) instead of Google Drive, use the `--drive` override flag:
```bash
# Force local workspace output writing:
!python src/training/train.py --config src/configs/resnet50.yaml --drive False
```

---

## 4. Multi-Model Pipeline Queueing (The "Fire-and-Forget" Scripts)

If you are training multiple models in a row, you can chain commands together so you don't have to monitor Colab and launch them manually one-by-one.

### 4.1 Simple Bash Chain Command
Connect commands using `&&` (ensures next model only trains if the previous finished successfully):
```bash
!python src/training/train.py --config src/configs/densenet121.yaml && \
python src/training/train.py --config src/configs/mobilenetv3_small.yaml && \
python src/training/train.py --config src/configs/efficientnet_b2.yaml && \
python src/training/train.py --config src/configs/efficientnet_b3.yaml && \
python src/training/train.py --config src/configs/convnext_tiny.yaml && \
python src/training/train.py --config src/configs/dinov2_small.yaml
```

### 4.2 Supervised Python Queue Pipeline (Recommended)
We created a custom orchestration script **[`scripts/run_experiments_queue.py`](file:///d:/Dental_Image_Classifier/scripts/run_experiments_queue.py)** that runs training for all remaining architectures **and automatically evaluates them on the hold-out test set** immediately after training, locating the dynamic timestamp checkpoint directory automatically!

Run this command inside Colab:
```bash
!python scripts/run_experiments_queue.py
```
* **Why this is superior**: With the simple bash chain, you would still have to manually find the timestamped checkpoint folder paths to run the evaluator. The Python script completely automates both training AND evaluation for all queued models in a single "fire-and-forget" command!

---

## 5. Post-Training Evaluation Commands

Evaluates a model's best saved weights on the hold-out test split (786 images). Generates per-class classification metrics, training history plots, normalized confusion matrix heatmaps, and failure predictions dump.

```bash
# Syntax: python src/evaluation/evaluator.py --config <config_path> --checkpoint <checkpoint_path>
!python src/evaluation/evaluator.py --config src/configs/resnet50.yaml --checkpoint /content/drive/MyDrive/dental_research/checkpoints/resnet50/resnet50_<timestamp>/best_model.pth
```
* *Note: Replace `<timestamp>` with your actual timestamp directory generated in Drive.*

---

## 6. Cross-Model Comparison & Paper Asset Generation

Once you have trained and evaluated your models, run the compiler tool to gather all results from the `experiments` directories and compile summary reports:

```bash
!python src/evaluation/compare_models.py
```

### Generated Paper Assets Locations:
* 📝 **LaTeX Table block**: `outputs/reports/model_comparison.tex` (ready to copy-paste directly into your LaTeX paper document!)
* 📝 **Markdown Report**: `outputs/reports/model_comparison.md`
* 📊 **Grouped Performance Chart**: `outputs/plots/model_comparison_bar.png`
* 🗺️ **Per-Class Comparative Heatmap**: `outputs/plots/per_class_f1_comparison.png`

---

## 7. Real-Time TensorBoard Dashboard
To open the live training dashboard inside Colab to monitor loss/accuracy decay while training runs:
```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/dental_research/logs/tensorboard
```

---

## 8. Stratified K-Fold Cross-Validation

To prove statistical stability and address the specific K-Fold requirements (K=2, 3, 5, 7, 9) in the research guidelines, we developed a dedicated cross-validation engine **[`scripts/run_kfold.py`](file:///d:/Dental_Image_Classifier/scripts/run_kfold.py)**.

### 8.1 Why K-Fold is Executed on the Champion Model:
Training all 8 architectures across all 5 values of K would require $8 \times 26 = 208$ separate deep learning training runs, which would immediately exceed Google Colab's T4 GPU usage quotas. In ML research, the standard best-practice is to **select the best-performing architecture (the "Champion")** and run the K-Fold sweep exclusively on it to demonstrate cross-validation robustness.

### 8.2 Execution Command
To run the stratified K-Fold cross-validation for the champion model (**Swin-Tiny**) and number of folds (e.g., K=5) in Colab:

```bash
# Syntax: python scripts/run_kfold.py --config <config_path> --k <number_of_folds>
!python scripts/run_kfold.py --config src/configs/swin_tiny.yaml --k 5
```

### 8.3 Sweeping K=2, 3, 5, 7, 9
To populate your exact paper requirements table, run the sweeps in a sequence:
```bash
!python scripts/run_kfold.py --config src/configs/swin_tiny.yaml --k 2 && \
python scripts/run_kfold.py --config src/configs/swin_tiny.yaml --k 3 && \
python scripts/run_kfold.py --config src/configs/swin_tiny.yaml --k 5 && \
python scripts/run_kfold.py --config src/configs/swin_tiny.yaml --k 7 && \
python scripts/run_kfold.py --config src/configs/swin_tiny.yaml --k 9
```

### 8.4 Generated K-Fold Reports Locations:
* 📝 **JSON Detail Report**: `outputs/kfold/swin_tiny_k<K>_report.json`
* 📝 **Consolidated Markdown Table**: `outputs/reports/kfold_metrics_table.md` (appends every completed sweep automatically, making it extremely easy to copy-paste the K-Fold values into your paper draft!)
