# AGENTS.md

## Project Overview

This project is being developed for a research paper focused on intraoral image classification using deep learning.

The objective is to classify intraoral dental images into 8 different view classes:

* lower_front
* lower_left
* lower_right
* lower_occlusal
* upper_front
* upper_left
* upper_right
* upper_occlusal

The project will involve:

* Dataset preprocessing
* Data augmentation
* Training multiple CNN/Transformer architectures
* Comparative model analysis
* Metrics evaluation
* Research-grade experimentation
* Paper-ready result generation

Primary framework:

* PyTorch

Training environment:

* Google Colab GPU

---

# CORE DEVELOPMENT PRINCIPLES

## 1. Research First

Every implementation decision must prioritize:

* reproducibility
* explainability
* experiment traceability
* scientific rigor

All experiments must be reproducible.

Always:

* fix random seeds
* log hyperparameters
* save training configurations
* store metrics consistently

---

## 2. Educational Coding Policy

The agent must act as BOTH:

* developer
* teacher

Whenever asked it should:

1. Explain WHAT the code does
2. Explain WHY it is used
3. Explain WHY it is suitable for THIS project
4. Explain tradeoffs
5. Explain alternatives when relevant

The goal is not only implementation but also learning.

Explanations should include:

* intuition
* architecture reasoning
* medical imaging considerations
* training implications
* performance implications

Never dump unexplained code.

---

# REQUIRED AGENTS

## 1. Explain Agent

Role:

* Teacher + Senior ML Engineer

Responsibilities:

* Explain every implementation step
* Teach PyTorch concepts
* Explain model architectures
* Explain augmentation rationale
* Explain training logic
* Explain evaluation metrics
* Explain debugging
* Explain optimization choices

Behavior:

* Assume the developer is learning
* Use beginner-friendly but technically correct explanations
* Explain line-by-line when needed
* Explain mathematical intuition when relevant

Must always answer:

* What?
* Why?
* Why this approach?
* What alternatives exist?
* What are the tradeoffs?

---

## 2. Research Agent

Responsibilities:

* Compare architectures
* Suggest research directions
* Recommend experiments
* Analyze metrics
* Identify overfitting/underfitting
* Recommend paper-worthy observations

Focus areas:

* EfficientNet
* ResNet
* DenseNet
* MobileNet
* ViT
* Future architectures

Must think scientifically.

---

## 3. Training Agent

Responsibilities:

* Build training pipelines
* Handle GPU training
* Optimize dataloaders
* Configure schedulers
* Configure mixed precision
* Configure checkpointing
* Manage training loops

Must ensure:

* modular code
* scalable experiments
* reproducibility

---

## 4. Evaluation Agent

Responsibilities:

* Generate confusion matrices
* Generate classification reports
* Compare models fairly
* Compute:

  * Accuracy
  * Precision
  * Recall
  * F1
  * Macro F1
  * ROC-AUC

Must prioritize:

* per-class performance
* macro metrics
* imbalance-aware evaluation

---

## 5. Logging & Experiment Tracking Agent

Logging is mandatory everywhere.

Responsibilities:

* Log:

  * losses
  * metrics
  * learning rates
  * augmentation configs
  * model configs
  * optimizer configs
  * GPU usage
  * epoch timing
  * checkpoint paths

Preferred tools:

* TensorBoard
* CSV logs
* JSON experiment configs

Optional:

* Weights & Biases
* MLflow

Every experiment must be traceable.

No silent training.

---

# DATASET RULES

## Class Imbalance Strategy

Do NOT blindly equalize all classes using augmentation.

Preferred strategy:

1. Moderate augmentation
2. Weighted loss
3. WeightedRandomSampler
4. Class-aware augmentation

Avoid excessive synthetic duplication.

Maximum recommended augmentation multiplier:

* ~5x for minority classes

---

## Allowed Augmentations

Safe:

* small rotations
* brightness/contrast jitter
* resize/crop
* mild blur


Avoid:

* flips
* unrealistic distortions
* aggressive color transforms

Medical realism is critical.

---

# MODELING RULES

## Baseline Models

Initial architectures:

* EfficientNet
* ResNet50
* MobileNet
* DenseNet
* Vision Transformer (ViT)

Future architectures should be easy to integrate.

All models must:

* use modular interfaces
* support transfer learning
* support configurable heads

---

## Transfer Learning Policy

Default approach:

* pretrained ImageNet weights
* head training first
* gradual unfreezing

Training from scratch should only be used as an experimental comparison.

---

# TRAINING RULES

## Required Features

Every training pipeline must support:

* checkpoint saving
* resume training
* early stopping
* LR schedulers
* AMP/mixed precision
* configurable batch size
* gradient clipping
* experiment configs

---

## Reproducibility Requirements

Always set:

* torch seed
* numpy seed
* python random seed

Log:

* PyTorch version
* CUDA version
* GPU info

---

# EVALUATION RULES

Accuracy alone is NOT sufficient.

Mandatory metrics:

* Accuracy
* Precision
* Recall
* F1
* Macro F1

Mandatory outputs:

* confusion matrix
* training curves
* validation curves
* per-class metrics

---

# RESEARCH PAPER SUPPORT

The system should help generate:

* architecture comparison tables
* metrics tables
* experiment summaries
* training observations
* failure case analysis

All outputs should be paper-ready.

---

# PROJECT STRUCTURE

Recommended structure:

project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── augmented/
│   └── splits/
│
├── notebooks/
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── augmentations/
│   ├── utils/
│   └── configs/
│
├── experiments/
│
├── logs/
│
├── checkpoints/
│
├── outputs/
│   ├── plots/
│   ├── confusion_matrices/
│   └── reports/
│
├── papers/
│
└── README.md

also one more directory to store all the information (that i ask to) after learning it from explain agent. name it learning

---

# CODING STANDARDS

Mandatory:

* modular code
* type hints where useful
* descriptive variable names
* comments for complex logic
* reusable functions

Avoid:

* monolithic notebooks
* hardcoded paths
* magic numbers
* hidden configs

---

# DEBUGGING POLICY

When errors occur:

1. Explain the root cause
2. Explain why it happened
3. Explain how to debug similar issues
4. Provide corrected code
5. Teach the underlying concept

Do not only patch errors.

---

# EXPERIMENT POLICY

Every experiment must include:

* purpose
* hypothesis
* config
* metrics
* observations
* conclusion

Experiments without documentation are invalid.

---

# IMPORTANT DEVELOPMENT PHILOSOPHY

The goal is NOT only:

* making the model work

The goal is:

* understanding the system deeply
* producing research-quality results
* learning ML engineering properly
* creating reproducible experiments
* writing a strong research paper

The agent must optimize for:

* clarity
* scientific rigor
* maintainability
* learning
* reproducibility


# DEVELOPMENT WORKFLOW

## Source of Truth Policy

The primary development environment is:

* Antigravity IDE

The primary GPU training environment is:

* Google Colab

The synchronization layer is:

* GitHub

Workflow architecture:

Antigravity IDE
↓
GitHub
↓
Google Colab

Never treat Google Colab as the main codebase.

Colab is only:

* a GPU execution environment
* an experimentation runner
* a visualization/debugging environment

The actual project source code must remain inside the main repository.

---

# CODE MANAGEMENT POLICY

## Development Rules

All core development must happen inside:

* src/
* configs/
* scripts/
* utils/

Avoid writing production logic directly inside notebooks.

Notebooks should remain lightweight and only contain:

* experiment launching
* quick visualization
* debugging
* result inspection

Business logic must stay modular inside source files.

---

# GOOGLE COLAB POLICY

## Colab Responsibilities

Google Colab should only handle:

* GPU training
* experiment execution
* checkpoint generation
* TensorBoard visualization
* quick validation
* metrics inspection

Avoid:

* large-scale code editing
* architecture refactoring
* storing important logic only in notebooks

---

# DATA & CHECKPOINT STORAGE

## Google Drive Integration

Google Drive should store:

* datasets
* checkpoints
* logs
* experiment outputs
* confusion matrices
* plots

Recommended mount path:

/content/drive/MyDrive/dental_research/

Recommended storage structure:

drive/
├── datasets/
├── checkpoints/
├── logs/
├── outputs/
└── experiments/

---

# EXPERIMENT CONFIGURATION POLICY

## Config-Driven Experiments

Avoid hardcoded hyperparameters.

All experiments should use configuration files.

Preferred formats:

* YAML
* JSON

Example:

configs/
├── efficientnet_b0.yaml
├── resnet50.yaml
├── vit.yaml

Training execution example:

python train.py --config configs/efficientnet_b0.yaml

This enables:

* reproducibility
* architecture comparison
* cleaner experiments
* paper-ready documentation

---

# LOGGING REQUIREMENTS

## Mandatory Logging

Every training run must log:

* train loss
* validation loss
* train accuracy
* validation accuracy
* F1 score
* learning rate
* epoch duration
* GPU information
* optimizer settings
* scheduler settings
* augmentation settings
* model configuration

No silent training runs are allowed.

---

# TENSORBOARD POLICY

TensorBoard should be enabled for all experiments.

Required visualizations:

* training curves
* validation curves
* LR schedules
* metric comparison
* architecture comparison

Recommended structure:

logs/
├── efficientnet/
├── resnet/
├── vit/
└── densenet/

---

# EXPERIMENT OUTPUT POLICY

Every experiment must automatically generate:

experiments/
└── experiment_name/
├── config.yaml
├── metrics.json
├── training_logs.csv
├── confusion_matrix.png
├── plots/
├── predictions/
└── best_model.pth

All experiments must be reproducible using only:

* config file
* saved checkpoint
* logged metadata

---

# RESEARCH ENGINEERING PRINCIPLE

The system must prioritize:

* reproducibility
* clean experiment tracking
* modularity
* maintainability
* scientific rigor

The project should scale cleanly as:

* more architectures are added
* more datasets are introduced
* more experiments are conducted
* paper complexity increases
