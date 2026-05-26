# Model Architecture Specifications — Comparative Study for Intraoral Dental Image Classification

> **Research Paper Reference**: Section 4 — Experimental Setup / Model Selection
> **Date of Compilation**: 2026-05-27
> **Task**: 8-class intraoral view classification
> **Input Dimensions**: 224 × 224 × 3 (RGB)
> **Output Classes**: 8 (lower_front, lower_left, lower_occlusal, lower_right, upper_front, upper_left, upper_occlusal, upper_right)
> **Framework**: PyTorch + timm (PyTorch Image Models)

---

## 1. Study Design Rationale

This comparative study evaluates **8 distinct architectures** spanning four fundamental paradigm families of modern computer vision. The selection is intentionally designed to enable rigorous cross-paradigm comparison for the research paper:

| Paradigm Family | Architectures Included | Core Mechanism |
| :--- | :--- | :--- |
| **Classical CNNs** | ResNet-50, DenseNet-121 | Residual connections, dense feature reuse |
| **Efficient / Mobile CNNs** | MobileNetV3-Small, EfficientNet-B2, EfficientNet-B3 | Depthwise separable convolutions, compound scaling |
| **Modern Hybrid CNNs** | ConvNeXt-Tiny | Modernized ConvNet with Transformer-inspired design |
| **Vision Transformers** | Swin Transformer Tiny, TransUNet (R50+ViT Hybrid), DINOv2-Small (ViT-S/14) | Self-attention, shifted windows, self-supervised pretraining |

This diversity allows the paper to make authoritative claims about:
- Whether lightweight models (MobileNet) can match heavyweight architectures for dental view classification
- Whether self-attention mechanisms (Transformers) provide advantages over convolution for this specific medical imaging task
- Whether self-supervised pretraining (DINOv2) outperforms supervised ImageNet pretraining
- The efficiency–accuracy tradeoff across parameter scales (2.5M → 105M)

---

## 2. Architecture-by-Architecture Detailed Specifications

---

### 2.1 ResNet-50 (Residual Network)

| Attribute | Specification |
| :--- | :--- |
| **Full Name** | Residual Network, 50 layers |
| **Paper** | He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016) |
| **Paradigm** | Classical CNN |
| **Parameters** | ~25.6M (total), ~23.5M (backbone) |
| **FLOPs** | ~4.1 GFLOPs |
| **Input Size** | 224 × 224 |
| **timm Model Name** | `resnet50.a1_in1k` |
| **Pretrained Weights** | ImageNet-1K (supervised) |
| **Feature Dim** | 2048 |
| **Classification Head** | Global Average Pooling → Linear(2048, 8) |
| **ImageNet-1K Top-1** | ~80.4% |

**Architecture Summary**: ResNet-50 introduced **skip (residual) connections** that allow gradient flow through identity shortcuts, solving the vanishing gradient problem in very deep networks. It consists of 4 stages with [3, 4, 6, 3] bottleneck blocks respectively. Each bottleneck block contains a 1×1 → 3×3 → 1×1 convolution sequence with batch normalization and ReLU activations.

**Why Include**: ResNet-50 is the **universal baseline** in image classification research. Nearly every medical imaging paper includes it. Its inclusion ensures our results are directly comparable to the broader literature.

**Relevance to Dental Imaging**: The hierarchical multi-scale feature extraction (from edges to textures to parts to objects) aligns well with dental view classification, where the model must recognize tooth arrangements, gum lines, and arch curvature at multiple spatial scales.

---

### 2.2 DenseNet-121 (Densely Connected Network)

| Attribute | Specification |
| :--- | :--- |
| **Full Name** | Densely Connected Convolutional Network, 121 layers |
| **Paper** | Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017) |
| **Paradigm** | Classical CNN |
| **Parameters** | ~8.0M |
| **FLOPs** | ~2.9 GFLOPs |
| **Input Size** | 224 × 224 |
| **timm Model Name** | `densenet121.ra_in1k` |
| **Pretrained Weights** | ImageNet-1K (supervised) |
| **Feature Dim** | 1024 |
| **Classification Head** | Global Average Pooling → Linear(1024, 8) |
| **ImageNet-1K Top-1** | ~75.6% |

**Architecture Summary**: DenseNet connects every layer to every other layer in a feed-forward fashion within each dense block. For a block with $L$ layers, there are $\frac{L(L+1)}{2}$ direct connections. This **dense connectivity pattern** encourages feature reuse, reduces the number of parameters compared to ResNet, and strengthens gradient flow. DenseNet-121 uses 4 dense blocks with [6, 12, 24, 16] layers and a growth rate of 32.

**Why Include**: DenseNet is significantly more parameter-efficient than ResNet (8M vs 25.6M) while achieving competitive accuracy. For a research paper, this establishes whether feature reuse via dense connections provides an advantage over residual shortcuts for dental view classification.

**Relevance to Dental Imaging**: The dense feature reuse is particularly beneficial for medical imaging tasks where subtle texture differences (enamel surface, gingival margin, dental material reflections) at multiple resolutions are critical for accurate classification.

---

### 2.3 MobileNetV3-Small

| Attribute | Specification |
| :--- | :--- |
| **Full Name** | MobileNet Version 3, Small variant |
| **Paper** | Howard et al., "Searching for MobileNetV3" (ICCV 2019) |
| **Paradigm** | Efficient / Mobile CNN |
| **Parameters** | ~2.5M |
| **FLOPs** | ~0.06 GFLOPs |
| **Input Size** | 224 × 224 |
| **timm Model Name** | `mobilenetv3_small_100.lamb_in1k` |
| **Pretrained Weights** | ImageNet-1K (supervised) |
| **Feature Dim** | 576 |
| **Classification Head** | Global Average Pooling → Linear(576, 1024) → Hardswish → Linear(1024, 8) |
| **ImageNet-1K Top-1** | ~67.7% |

**Architecture Summary**: MobileNetV3 uses **depthwise separable convolutions** (which factorize a standard convolution into a depthwise and pointwise step, reducing computation by a factor of ~8–9x), combined with **Squeeze-and-Excitation (SE) attention blocks** and **hard-swish activations** found via Neural Architecture Search (NAS). The "Small" variant is optimized for low-resource deployment.

**Why Include**: MobileNetV3-Small establishes the **lower bound of model capacity** in our study. If a 2.5M parameter model can achieve competitive dental view classification accuracy, it implies the task has limited visual complexity and does not require massive architectures. This has direct implications for clinical deployment on edge devices (dental chair tablets, mobile phones).

**Relevance to Dental Imaging**: Dental clinics increasingly use tablet-based or phone-based image capture. If MobileNet performs well, it enables real-time on-device inference without cloud dependency — a significant practical contribution.

---

### 2.4 EfficientNet-B2

| Attribute | Specification |
| :--- | :--- |
| **Full Name** | EfficientNet, B2 scaling variant |
| **Paper** | Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs" (ICML 2019) |
| **Paradigm** | Efficient CNN |
| **Parameters** | ~9.1M |
| **FLOPs** | ~1.0 GFLOPs |
| **Input Size** | 260 × 260 (native), will resize to 224 × 224 |
| **timm Model Name** | `efficientnet_b2.ra_in1k` |
| **Pretrained Weights** | ImageNet-1K (supervised) |
| **Feature Dim** | 1408 |
| **Classification Head** | Global Average Pooling → Dropout(0.3) → Linear(1408, 8) |
| **ImageNet-1K Top-1** | ~80.6% |

**Architecture Summary**: EfficientNet uses a **compound scaling method** that uniformly scales network width, depth, and resolution using a set of fixed scaling coefficients. The base architecture (B0) is discovered via NAS and uses **Mobile Inverted Bottleneck Convolutions (MBConv)** with SE attention. B2 applies a moderate compound scaling factor ($\phi = 1$, width × 1.1, depth × 1.2, resolution × 1.3).

**Why Include**: EfficientNet-B2 provides a sweet spot between model size and accuracy. With only 9.1M parameters, it matches or exceeds ResNet-50's ImageNet accuracy. Comparing B2 with B3 directly demonstrates the impact of compound scaling on dental classification performance.

**Relevance to Dental Imaging**: The SE attention blocks in MBConv are particularly relevant because they allow the network to dynamically re-weight channel responses, potentially learning to emphasize dental-specific color channels (gum pink, tooth enamel white, filling material grey).

---

### 2.5 EfficientNet-B3

| Attribute | Specification |
| :--- | :--- |
| **Full Name** | EfficientNet, B3 scaling variant |
| **Paper** | Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs" (ICML 2019) |
| **Paradigm** | Efficient CNN |
| **Parameters** | ~12.2M |
| **FLOPs** | ~1.8 GFLOPs |
| **Input Size** | 300 × 300 (native), will resize to 224 × 224 |
| **timm Model Name** | `efficientnet_b3.ra2_in1k` |
| **Pretrained Weights** | ImageNet-1K (supervised) |
| **Feature Dim** | 1536 |
| **Classification Head** | Global Average Pooling → Dropout(0.3) → Linear(1536, 8) |
| **ImageNet-1K Top-1** | ~82.0% |

**Architecture Summary**: B3 applies a higher compound scaling factor ($\phi = 1.2$) than B2: width × 1.2, depth × 1.4, resolution × 1.6. This yields a wider, deeper network with higher native resolution. The core MBConv + SE architecture remains identical to B2.

**Why Include**: The B2 vs B3 comparison within the same architecture family enables a clean, controlled experiment to answer: *"Does increasing model capacity (width, depth, resolution) via compound scaling improve dental view classification, or does it lead to overfitting on our ~5,200 image dataset?"*

**Relevance to Dental Imaging**: The higher capacity of B3 may capture finer-grained dental features (individual tooth morphology, subtle occlusal surface patterns), but the risk of overfitting on our moderately sized dataset is a real concern that the paper should discuss.

---

### 2.6 ConvNeXt-Tiny

| Attribute | Specification |
| :--- | :--- |
| **Full Name** | A ConvNet for the 2020s, Tiny variant |
| **Paper** | Liu et al., "A ConvNet for the 2020s" (CVPR 2022) |
| **Paradigm** | Modern Hybrid CNN |
| **Parameters** | ~28.6M |
| **FLOPs** | ~4.5 GFLOPs |
| **Input Size** | 224 × 224 |
| **timm Model Name** | `convnext_tiny.in12k_ft_in1k` |
| **Pretrained Weights** | ImageNet-12K → ImageNet-1K (fine-tuned) |
| **Feature Dim** | 768 |
| **Classification Head** | Global Average Pooling → LayerNorm → Linear(768, 8) |
| **ImageNet-1K Top-1** | ~82.9% |

**Architecture Summary**: ConvNeXt systematically "modernizes" a standard ResNet by adopting design choices from Vision Transformers: larger kernel sizes (7×7 depthwise convolutions instead of 3×3), inverted bottleneck structure, fewer activation functions (GELU instead of ReLU), LayerNorm instead of BatchNorm, and a patchify stem (4×4 non-overlapping convolutions). The result is a pure convolutional architecture that matches or exceeds Swin Transformer accuracy.

**Why Include**: ConvNeXt is the critical "bridge" model in our study. It uses **only convolutions** (no self-attention) but is designed with Transformer-inspired principles. If ConvNeXt outperforms Swin Transformer, it suggests that the performance advantage of Transformers comes from their training recipes and design macros, not from self-attention itself — a significant research finding.

**Relevance to Dental Imaging**: The larger 7×7 receptive fields in ConvNeXt may better capture the broad spatial layout of the dental arch, which spans the full image width, compared to traditional 3×3 convolutions that process local textures.

---

### 2.7 Swin Transformer Tiny

| Attribute | Specification |
| :--- | :--- |
| **Full Name** | Shifted Window Transformer, Tiny variant |
| **Paper** | Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (ICCV 2021) |
| **Paradigm** | Vision Transformer |
| **Parameters** | ~28.3M |
| **FLOPs** | ~4.5 GFLOPs |
| **Input Size** | 224 × 224 |
| **timm Model Name** | `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k` |
| **Pretrained Weights** | ImageNet-22K → ImageNet-1K (fine-tuned) |
| **Feature Dim** | 768 |
| **Patch Size** | 4 × 4 |
| **Window Size** | 7 × 7 |
| **Attention Heads** | [3, 6, 12, 24] across 4 stages |
| **Classification Head** | Global Average Pooling → Linear(768, 8) |
| **ImageNet-1K Top-1** | ~81.2% |

**Architecture Summary**: Swin Transformer computes self-attention within **non-overlapping local windows** (7×7 patches), and then **shifts** these windows by half the window size in alternating layers to enable cross-window information flow. This creates a hierarchical feature map (like CNNs) while maintaining linear computational complexity with respect to image size $O(n)$, unlike standard ViT which has quadratic complexity $O(n^2)$.

**Why Include**: Swin Transformer is the dominant hierarchical Vision Transformer for image classification. Its shifted window mechanism is fundamentally different from both standard convolutions and global self-attention. Comparing it against ConvNeXt (same parameter count, same FLOPs, different mechanism) provides a clean apples-to-apples Transformer vs CNN comparison.

**Relevance to Dental Imaging**: The local window attention may naturally align with dental image structure — individual teeth occupy local spatial regions, while the overall arch arrangement requires cross-region (shifted window) communication. This architectural inductive bias could be beneficial for dental view recognition.

---

### 2.8 TransUNet (R50-ViT Hybrid Encoder for Classification)

| Attribute | Specification |
| :--- | :--- |
| **Full Name** | TransUNet Hybrid Encoder (ResNet-50 + Vision Transformer) |
| **Paper** | Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation" (arXiv 2021) |
| **Paradigm** | CNN-Transformer Hybrid |
| **Parameters** | ~105M (full R50-ViT-L/16 encoder) |
| **FLOPs** | ~15+ GFLOPs |
| **Input Size** | 224 × 224 |
| **Implementation** | Custom (based on official TransUNet repository) |
| **Pretrained Weights** | ViT-L/16 + ResNet-50 hybrid (ImageNet-21K) |
| **Feature Dim** | 768 (ViT output CLS token) |
| **Classification Head** | CLS token → LayerNorm → Linear(768, 8) |
| **ImageNet-1K Top-1** | N/A (originally designed for segmentation) |

**Architecture Summary**: TransUNet was originally proposed for medical image segmentation (U-Net with Transformer encoder). The **encoder portion** consists of a ResNet-50 backbone that extracts spatial feature maps, which are then tokenized and fed into a 12-layer Vision Transformer. For our classification task, we **discard the U-Net decoder** and extract the CLS token from the Transformer output as a global image representation.

**Why Include**: TransUNet is specifically designed for medical imaging. Its hybrid CNN+Transformer encoder combines the local feature extraction strength of ResNet with the global context modeling capability of ViT. Including it allows us to test whether a **medical-imaging-native architecture** outperforms general-purpose architectures (ResNet, EfficientNet, Swin) on our dental dataset.

**Relevance to Dental Imaging**: The R50 backbone preserves fine spatial details (tooth boundaries, filling edges), while the ViT layers capture the global arrangement of the dental arch. This dual-scale feature extraction is architecturally motivated for clinical image analysis.

> **Implementation Note**: Since TransUNet is primarily a segmentation model, we will implement a custom `TransUNetEncoder` class that loads the pretrained hybrid weights and adds a classification head. The U-Net decoder is not used.

---

### 2.9 DINOv2-Small (ViT-S/14)

| Attribute | Specification |
| :--- | :--- |
| **Full Name** | Self-Distillation with No Labels v2, ViT-Small with patch size 14 |
| **Paper** | Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (TMLR 2024) |
| **Paradigm** | Self-Supervised Vision Transformer |
| **Parameters** | ~22.1M |
| **FLOPs** | ~4.6 GFLOPs (at 224×224) |
| **Input Size** | 224 × 224 (flexible; native 518×518) |
| **timm Model Name** | `vit_small_patch14_dinov2.lvd142m` |
| **Pretrained Weights** | LVD-142M (142M curated images, self-supervised) |
| **Feature Dim** | 384 |
| **Patch Size** | 14 × 14 |
| **Attention Heads** | 6 |
| **Transformer Layers** | 12 |
| **Classification Head** | CLS token → Linear(384, 8) |
| **ImageNet-1K Top-1** | ~81.1% (linear probe) |

**Architecture Summary**: DINOv2 is a **self-supervised Vision Transformer** that learns visual representations without any human labels. It uses a self-distillation framework where a student network learns to match the output of an exponential moving average (EMA) teacher network. The model was pretrained on LVD-142M, a curated dataset of 142 million images collected from diverse sources. The resulting features are remarkably general-purpose and can be used as frozen feature extractors or fine-tuned.

**Why Include**: DINOv2 represents a fundamentally different pretraining paradigm — **self-supervised** rather than **supervised** (ImageNet labels). If DINOv2 outperforms supervised pretrained models, it suggests that the visual features learned through self-distillation are more transferable to medical imaging domains where labeled data is scarce. This is a highly relevant finding for the research paper.

**Relevance to Dental Imaging**: Self-supervised features are known to capture rich structural and textural representations that generalize well to out-of-distribution domains. Since dental intraoral images are visually very different from ImageNet's natural photographs, DINOv2's domain-agnostic features may transfer more robustly than supervised ImageNet features which are biased toward natural image statistics.

---

## 3. Comparative Summary Table

| # | Architecture | Family | Params | FLOPs | Feature Dim | Pretrain Data | Pretrain Method | timm Name |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| 1 | ResNet-50 | Classical CNN | 25.6M | 4.1G | 2048 | ImageNet-1K | Supervised | `resnet50.a1_in1k` |
| 2 | DenseNet-121 | Classical CNN | 8.0M | 2.9G | 1024 | ImageNet-1K | Supervised | `densenet121.ra_in1k` |
| 3 | MobileNetV3-S | Mobile CNN | 2.5M | 0.06G | 576 | ImageNet-1K | Supervised | `mobilenetv3_small_100.lamb_in1k` |
| 4 | EfficientNet-B2 | Efficient CNN | 9.1M | 1.0G | 1408 | ImageNet-1K | Supervised | `efficientnet_b2.ra_in1k` |
| 5 | EfficientNet-B3 | Efficient CNN | 12.2M | 1.8G | 1536 | ImageNet-1K | Supervised | `efficientnet_b3.ra2_in1k` |
| 6 | ConvNeXt-Tiny | Modern CNN | 28.6M | 4.5G | 768 | IN-12K → IN-1K | Supervised | `convnext_tiny.in12k_ft_in1k` |
| 7 | Swin-Tiny | Vision Transformer | 28.3M | 4.5G | 768 | IN-22K → IN-1K | Supervised | `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k` |
| 8 | TransUNet | Hybrid CNN+ViT | ~105M | ~15G | 768 | ImageNet-21K | Supervised | Custom implementation |
| 9 | DINOv2-Small | Self-Sup. ViT | 22.1M | 4.6G | 384 | LVD-142M | Self-supervised | `vit_small_patch14_dinov2.lvd142m` |

---

## 4. Transfer Learning Protocol

All models (except TransUNet which requires custom handling) follow a **uniform two-phase transfer learning protocol** to ensure fair comparison:

### Phase 1: Head-Only Training (Feature Extraction)
- **Freeze**: Entire pretrained backbone (all convolutional/attention layers)
- **Train**: Only the newly initialized classification head
- **Epochs**: 10
- **Purpose**: Allow the classification head to calibrate to the dental domain's feature distribution without disturbing learned visual representations
- **Learning Rate**: Higher (1e-3), since only the head is being optimized

### Phase 2: Full Fine-Tuning (Gradual Unfreezing)
- **Unfreeze**: Progressively unfreeze backbone layers starting from the deepest
- **Train**: All parameters
- **Epochs**: 30–50 (with early stopping)
- **Purpose**: Adapt the pretrained visual features to dental-specific patterns
- **Learning Rate**: Lower (1e-4 to 1e-5), with cosine annealing schedule

---

## 5. Unified Training Configuration

To ensure fair and reproducible comparison, all models share the following training hyperparameters:

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Input Size** | 224 × 224 | Standard size supported by all 8 architectures |
| **Batch Size** | 32 | Fits comfortably in Colab GPU memory (T4/A100) |
| **Optimizer** | AdamW | Decoupled weight decay; standard for both CNNs and Transformers |
| **Weight Decay** | 0.01 | Regularization to prevent overfitting on ~5K images |
| **Base LR** | 1e-3 (head), 1e-4 (fine-tune) | Standard transfer learning rates |
| **LR Scheduler** | Cosine Annealing with Warmup | Smooth decay; prevents sharp LR drops |
| **Warmup Epochs** | 5 | Stabilizes early training |
| **Loss Function** | Cross-Entropy with class weights | Handles residual imbalance (684 vs 650) |
| **Sampler** | WeightedRandomSampler | Ensures balanced mini-batches per epoch |
| **Mixed Precision** | AMP (FP16) | 2x speedup on modern GPUs |
| **Early Stopping** | Patience = 7 epochs | Prevents wasted compute on overfitting runs |
| **Gradient Clipping** | max_norm = 1.0 | Stabilizes Transformer training |
| **Random Seed** | 42 | Full reproducibility |
| **Data Split** | 70% train / 15% val / 15% test | Stratified by class |
| **Online Augmentations** | Normalize (ImageNet stats) + RandomResizedCrop | Standard real-time augmentations during training |

---

## 6. Research Hypotheses

The following hypotheses will be tested through our comparative experiments:

1. **H1 (Capacity vs Overfitting)**: Models with >25M parameters (ResNet-50, ConvNeXt, Swin) will achieve higher peak validation accuracy than smaller models, but are more susceptible to overfitting on our ~5K image dataset.

2. **H2 (EfficientNet Scaling)**: EfficientNet-B3 will outperform B2 due to increased width and depth, but the improvement margin will be smaller than on ImageNet due to the domain gap.

3. **H3 (CNN vs Transformer)**: Swin Transformer and ConvNeXt (same parameter count, same FLOPs) will achieve similar accuracy, suggesting that the Transformer's advantage comes from its training recipe rather than self-attention itself.

4. **H4 (Self-Supervised Transfer)**: DINOv2-Small will provide competitive or superior classification accuracy compared to supervised ImageNet-pretrained models of similar size, validating the hypothesis that self-supervised representations transfer more robustly to medical imaging domains.

5. **H5 (Medical Hybrid)**: TransUNet's hybrid CNN+ViT encoder, designed for medical imaging, will achieve strong per-class recall on challenging view classes (upper_right, upper_front) due to its dual local-global feature extraction.

6. **H6 (Edge Deployment)**: MobileNetV3-Small will achieve >85% accuracy despite having 10–40x fewer parameters than the largest models, demonstrating that dental view classification can be deployed on resource-constrained clinical hardware.
