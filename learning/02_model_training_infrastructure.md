# Lesson 02 — Model Training Infrastructure & Two-Phase Fine-Tuning

> **Role**: Explain Agent (ML Engineer + Educator)  
> **Topic**: Training Infrastructure, Configurations, Epoch Counts, and Code Layout for the Comparative Multi-Model Study  
> **Target Audience**: Dental Image Classifier Research Paper Reference

This document explains **what** our training system does, **why** it was designed this way, **how** the configurations operate, **where** the code lives, and the core **tradeoffs and alternatives** associated with transfer learning on medical/dental datasets.

---

## 1. Where the Code Lives (The Architecture Layout)

To maintain a research-grade codebase, we strictly separate configurations, dataloaders, model construction, and execution loops. Below is the mapping of your training code files:

| Folder / File Path | Module | What It Does (The "What") | Why It is Essential |
| :--- | :--- | :--- | :--- |
| **[`src/training/train.py`](file:///d:/Dental_Image_Classifier/src/training/train.py)** | **Training Runner (CLI)** | The main entrance. Parses arguments, loads YAML configurations, handles Google Colab/Drive output paths redirection, and initializes custom normalized class-weighted Cross-Entropy loss. | Coordinates all sub-components. Prevents hardcoded paths, making running experiments on Colab seamless. |
| **[`src/training/trainer.py`](file:///d:/Dental_Image_Classifier/src/training/trainer.py)** | **Training Engine** | Defines the `Trainer` class. Coordinates batch training, validation loops, learning rate updates, AMP (FP16) scaling, gradient clipping, early stopping, and checkpointing. | Decouples the mathematical training loop from configuration parsing, making training execution clean and modular. |
| **[`src/training/schedulers.py`](file:///d:/Dental_Image_Classifier/src/training/schedulers.py)** | **LR Scheduler** | Implements the Warmup + Cosine Annealing learning rate schedule. | Prevents sudden gradient changes during the early epochs and guarantees smooth convergence. |
| **[`src/models/model_factory.py`](file:///d:/Dental_Image_Classifier/src/models/model_factory.py)** | **Model Factory** | Creates backbones using `timm`, swaps the default ImageNet linear layer with a custom 8-class linear head, and controls layer-freezing/unfreezing. | Single control interface for all 8 architectures. Guarantees fair comparisons. |
| **[`src/datasets/data_utils.py`](file:///d:/Dental_Image_Classifier/src/datasets/data_utils.py)** | **Dataloader Factory** | Maps training/validation/test datasets, applies **no-flip transforms**, and implements `WeightedRandomSampler`. | Resolves minority-class representation on a batch level, ensuring gradients are not biased. |
| **[`src/configs/*.yaml`](file:///d:/Dental_Image_Classifier/src/configs/)** | **Hyperparameters** | Contains 8 model-specific configuration blueprints. | Guarantees full reproducibility. You can rerun any experiment simply by supplying its YAML configuration. |

---

## 2. What are the YAML Files?

Instead of hardcoding batch sizes, learning rates, or model paths in Python code, **every parameter** is kept in a configuration YAML. Here is the blueprint using `resnet50.yaml` as an example:

```yaml
model:
  name: "resnet50"                     # Local identifier used for folder naming
  timm_name: "resnet50.a1_in1k"        # Exact model identifier in PyTorch Image Models (timm)
  pretrained: true                     # Default transfer learning strategy
  num_classes: 8                       # Core dental view classes count
  
training:
  seed: 42                             # Strict random seed for reproducibility
  batch_size: 32                       # Balanced for Colab T4 GPU VRAM limits
  phase1_epochs: 10                    # Calibrating head while backbone is frozen
  phase2_epochs: 40                    # Global fine-tuning
  phase1_lr: 1e-3                      # Faster rate for untrained random head weights
  phase2_lr: 1e-4                      # Slow, conservative rate for pre-trained weights
  weight_decay: 0.01                   # L2 regularisation to avoid overfitting
  warmup_epochs: 5                     # Gradual learning rate growth period
  early_stopping_patience: 7           # Stop training if validation accuracy plateaus
  gradient_clip_max_norm: 1.0          # Stabilises gradient updates
  amp: true                            # Automated Mixed Precision (speeds up T4 by ~2x)

data:
  data_dir: "data/augmented"           # Location of data
  split_ratios: [0.70, 0.15, 0.15]     # 70% Train, 15% Val, 15% Test
  input_size: 224                      # Receptive field width/height
  num_workers: 2                       # Parallel loading threads

paths:
  checkpoint_dir: "checkpoints"        # Weights path
  log_dir: "logs"                      # Logging path
  experiment_dir: "experiments"        # Experiment summary outputs path
```

---

## 3. How the Model Training Takes Place (The Training Protocol)

Our system uses a **Uniform Two-Phase Transfer Learning Protocol** to compare all 8 CNN and Transformer paradigms fairly:

```
                  ┌───────────────────────────────┐
                  │ 1. LOAD PRE-TRAINED BACKBONE  │
                  │   (e.g., ImageNet Weights)    │
                  └───────────────┬───────────────┘
                                  │
                                  v
                  ┌───────────────────────────────┐
                  │ 2. REPLACE CLASSIFIER HEAD    │
                  │   (Linear: FeatureDim ➔ 8)   │
                  └───────────────┬───────────────┘
                                  │
                                  v
 ┌────────────────────────────────┴──────────────────────────────┐
 │ PHASE 1: HEAD-ONLY FEATURE EXTRACTION                         │
 │ - Backbone: FROZEN (No weight updates)                        │
 │ - Head: TRAINABLE (Optimized at high LR = 1e-3)               │
 │ - Schedule: Linear Warmup ➔ Cosine Annealing                  │
 │ - Epochs: 10 Epochs                                           │
 └────────────────────────────────┬──────────────────────────────┘
                                  │
                  Transition: Dynamic Unfreezing
                  - All parameters set requires_grad = True
                  - Re-instantiate Optimizer with low LR = 1e-4
                  - Re-instantiate Scheduler
                                  │
                                  v
 ┌────────────────────────────────┴──────────────────────────────┐
 │ PHASE 2: GLOBAL MODEL FINE-TUNING                             │
 │ - Backbone: UNPROZEN (All weights adapting to dental views)   │
 │ - Head: TRAINABLE (Refining predictions at low LR = 1e-4)     │
 │ - Schedule: Linear Warmup ➔ Cosine Annealing                  │
 │ - Epochs: 40 Epochs (Early Stopping Enabled, Patience = 7)   │
 └───────────────────────────────────────────────────────────────┘
```

### Phase 1: Head-Only Calibration (10 Epochs)
* **What**: We freeze all convolutional/attention weights in the backbone. Only the newly swapped, randomly initialized classification head is trained.
* **Why**: The pre-trained features (ImageNet representations) are highly expressive but the new head weights are completely random. If we trained the entire network together from epoch 1 with standard learning rates, the massive backpropagated gradients from the random head would **destroy (wash out)** the rich pre-trained features in the backbone. This is known as **catastrophic representation destruction**.
* **Duration**: Configured for **10 Epochs** in all configurations.
* **Learning Rate**: Higher (**1e-3**), since we are only adjusting a single linear layer's weights.

### Transition: Dynamic Re-instantiation
* **What**: When moving from Phase 1 to Phase 2, we call `unfreeze_all_parameters()`. But crucially, we **must destroy and recreate** the optimizer and learning rate scheduler.
* **Why**: The Phase 1 optimizer *only* tracked head parameters. If we didn't recreate the optimizer, the backbone layers would remain frozen because the optimizer's parameter state wouldn't contain them! Additionally, we reset the learning rate down by a factor of 10.

### Phase 2: Global Fine-Tuning (40 Epochs)
* **What**: We unfreeze all layers of the model. All parameters are updated as the model processes dental images.
* **Why**: The pre-trained weights are adapted to dental-specific features (gum margins, tooth curvature, clinical tooth textures) rather than generic natural textures.
* **Duration**: Configured for **40 Epochs** max, but controlled by **Early Stopping (Patience = 7)**. If the validation accuracy does not improve for 7 consecutive epochs, training terminates to prevent overfitting.
* **Learning Rate**: Conservative (**1e-4**), keeping weight updates small to protect the baseline features.

---

## 4. Academic Justifications (What? Why? Tradeoffs? Alternatives?)

### Q1: Why use Transfer Learning instead of training from scratch?
* **Why**: Our cleaned, augmented dataset contains 5,234 images. Deep learning models (especially heavy architectures like Swin Transformer or ResNet-50) require hundreds of thousands of images to generalize effectively from scratch. Training from scratch on a small dataset leads to severe **overfitting** (where the model memorizes the training data but fails to classify new test images).
* **Tradeoff**: Pre-trained ImageNet weights introduce a **domain gap** (ImageNet has dogs, cars, and trees; our dataset has teeth and gums). However, visual foundations (edges, lighting shadows, textures, shapes) transfer robustly across domains.
* **Alternative**: Training from scratch. We will include MobileNetV3 trained from scratch in our research discussion as a comparison, proving that pre-training drastically accelerates convergence and boosts final accuracy.

### Q2: Why use a Two-Phase fine-tuning protocol?
* **Why**: It bridges the domain gap safely. Phase 1 lets the new classifier establish boundary boundaries based on fixed representations. Phase 2 adapts the actual feature-extractor filters to dental textures.
* **Tradeoff**: Training takes slightly longer because we partition the schedule. However, it significantly boosts final validation accuracy (usually by +3% to +8% compared to single-phase training) and guarantees training stability.
* **Alternative**: Single-Phase Joint Fine-tuning (training everything from epoch 1 at a very low learning rate). This works but converges slower and is highly sensitive to the initial head initialization.

### Q3: Why use both WeightedRandomSampler and Weighted Loss?
* **Why (Double-Layer Imbalance Protection)**: Although our dataset is balanced (650 per class), minor imbalances remain (e.g., `lower_left` has 684 clean unique images and was not downsampled). Additionally, during stratified splits, minor imbalances can manifest in batches. 
  * `WeightedRandomSampler` adjusts batch sampling frequency.
  * `Weighted Cross-Entropy Loss` scales loss gradients so minority classification errors are penalized more heavily.
* **Why this approach**: Under-represented dental views (which may have subtler visual differences) get amplified, guaranteeing high per-class Recall and high Macro F1-scores.
* **Tradeoff**: Adds minor computation, but essential for medical paper benchmarks where accuracy alone is not a valid proof of performance.
* **Alternative**: Blindly oversampling images offline (which would introduce exact duplicate copies in validation/test splits, causing severe **data leakage** and invalidating paper scientific rigor).
