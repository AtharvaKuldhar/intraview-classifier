# Clinical Image Augmentation and Dataset Standardization Report

> **Research Paper Reference**: Section 3.2 — Dataset Preprocessing and Balancing
> **Date of Generation**: 2026-05-27
> **Dataset Version**: v1.1-balanced
> **Primary Framework**: PyTorch 2.0+ & Albumentations 2.0.8

---

## Abstract
This report details the systematic data engineering pipeline designed to address severe class imbalance and data redundancy in intraoral photographs collected for view-based classification. Beginning with an initial raw collection of 3,950 images across 8 view classes, an MD5 cryptographic checksum analysis identified 967 within-class duplicates and 4 cross-class label conflicts. 

Following absolute deduplication, a class-aware, tier-based offline augmentation pipeline was executed using the OpenCV-based `albumentations` library. By applying deterministic, medically-safe transformations (rotation, color/contrast adjustment, Gaussian blur, and localized zooming) while strictly avoiding labels-contaminating flips, we generated 2,259 unique augmented instances. This produced a highly standardized, fully balanced dataset comprising **5,234 images** with an imbalance ratio of **1.05x** (down from 7.73x). Verification confirmed zero image corruptions and absolute uniqueness (0 duplicate hashes) in the final cohort.

---

## 1. Methodology & Data Integrity Pipeline

To guarantee scientific reproducibility and maintain the clinical integrity of the dataset, the preprocessing pipeline was structured into three disjoint phases:

```
                  +-----------------------------------+
                  |      Phase 1: Raw Dataset         |
                  |     (3,950 images collected)      |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  MD5 Hashing & Deduplication      |
                  |  - Removes 967 within-class dupes |
                  |  - Rejects 4 cross-class dupes    |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    Phase 2: Standardized Clean    |
                  |     (2,975 unique images)         |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Class-Aware Offline Augmentation  |
                  |  - Albumentations tier pipeline   |
                  |  - Standardized to 224x224 px     |
                  |  - Dynamic MD5 uniqueness filter  |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |     Phase 3: Final Cohort         |
                  |   (5,234 balanced images, 0 dup)  |
                  +-----------------------------------+
```

### 1.1 Phase 1: Cryptographic Deduplication
Clinical image collection is prone to duplication due to repeated backup synchronizations, folder merges, or manual export redundancies. Hashing images is the only mathematically rigorous method to catch exact duplicates.
- **Within-Class duplicates**: Detected by matching identical MD5 checksums within a single view class folder. Out of 3,950 files, 967 were exact replicas.
- **Cross-Class duplicates**: Instances where the exact same photograph was associated with conflicting labels (e.g., filed under both `lower_front` and `lower_left`). In machine learning, label noise of this nature degrades gradient calculations. Because the correct label could not be verified programmatically, all 8 file occurrences of the 4 cross-class duplicate images were **completely eliminated** from the dataset.

### 1.2 Phase 2: Structural Standardization
All clean unique images copied to the intermediate `data/processed/` directory were standardized. To ensure consistent dimensions for convolutional neural network (CNN) and Vision Transformer (ViT) input stages, both original and augmented files were downsampled/upsampled using bilinear interpolation to a fixed shape of **224 × 224 pixels** with 3 color channels (RGB).

### 1.3 Phase 3: Uniqueness Enforced Data Augmentation
To prevent the model from memorizing unaltered images, we implemented an in-memory hashing check in `augment.py`. If a set of randomly drawn augmentation parameters resulted in an output image that matched the MD5 hash of its original or any previously generated augmented image (which can happen when random trigger probabilities `p` do not fire), the system **automatically discarded** the output, incremented the random seed, and retried. This guaranteed 100% unique instances in the final augmented cohort.

---

## 2. Clinical and Medical Justifications for Transform Selections

Medical imaging datasets require highly strict augmentation policies. Arbitrary transformations that are common in natural image classification (e.g., ImageNet) can corrupt anatomical labels or represent physical scenarios that are clinically impossible.

### 2.1 Approved and Justified Transforms

| Transform Type | Albumentations Class | Clinical & Medical Rationale |
| :--- | :--- | :--- |
| **Small Rotation** | `A.Rotate` | Simulates natural variations in the patient's head tilt or slight alignment deviations in the intraoral camera angle during capture. Restricted to a maximum of $\pm15^\circ$ to maintain a realistic vertical horizon. |
| **Exposure & Contrast adjustment** | `A.RandomBrightnessContrast` | Essential for simulating variations in intraoral flash intensities (Ring Flash vs Point Flash), ambient dental chair overhead lighting, and varying sensor sensitivities between different dental clinics. |
| **Localized Zoom & Crop** | `A.RandomResizedCrop` | Simulates varying focal distances and physical zoom levels. Since a clinician might position the camera slightly closer or further from the dental arch, mild random cropping (constrained between 85% and 100% of original area) simulates these zoom discrepancies. |
| **Tissue Saturation** | `A.HueSaturationValue` | Standardizes minor color richness variations due to different camera manufacturers' default processing. Critically, the **Hue shift is locked at 0** because gums must remain pink/red and teeth must remain white/yellow; shifting the hue spectrum would create clinically nonsensical green or blue tissues. |
| **Lens Defocus / Blur** | `A.GaussianBlur` | Simulates minor motion blur or lens focus discrepancies, which are extremely common due to patient movement or quick captures inside the oral cavity. |
| **Aperture / Enamel Sharpening** | `A.Sharpen` | Accentuates details such as enamel borders and crack lines, simulating high-end clinical macro-lenses. |

### 2.2 Explicitly Banned Transforms ( label-contaminating / physically impossible )

- **Horizontal Flips (`A.HorizontalFlip`)**: **STRICTLY PROHIBITED**. Flipping an image horizontally would transform a "Lower Left View" into a "Lower Right View". This would directly contaminate class labels and train the network on false features.
- **Vertical Flips (`A.VerticalFlip`)**: **STRICTLY PROHIBITED**. Vertically flipping an image would swap the lower arch with the upper arch, violating anatomical gravity and clinical reality.
- **Elastic Deformations (`A.ElasticTransform` / `A.GridDistortion`)**: **STRICTLY PROHIBITED**. Warping tooth boundaries creates anatomical shapes that are physically impossible in human dentition and could mimic pathological deformities (e.g., severe bone loss or developmental anomalies) that the model should not learn as normal features.
- **Coarse Dropout / CutOut**: **STRICTLY PROHIBITED**. Occluding random sections of the teeth could block key diagnostic teeth (like the central incisors or first molars) that define the specific view class.
- **Channel Shuffling**: **STRICTLY PROHIBITED**. Swapping R, G, B channels would destroy color consistency, which is vital for identifying dental materials (amalgams, gold crowns, composite fillings) and gingival health.

---

## 3. Tiered Augmentation Specification

To avoid over-augmenting classes that already have sufficient unique samples, we used a **Class-Aware Tier System**. The intensity of transformations scales inversely with the size of the unique dataset in each class, adhering to the 5x safety cap (maximum multiplier of 4.48x applied to `upper_right`).

```
                    Unique Count           Augmentation Tier
                    ------------------------------------------
                    >= 650                 Tier 0 (None)
                    500 - 649              Tier 1 (Very Light)
                    250 - 499              Tier 2 (Moderate)
                    150 - 249              Tier 3 (Mod-Aggressive)
                    < 150                  Tier 4 (Aggressive)
```

### 3.1 Detailed Parameters by Tier

#### Tier 0: No Offline Augmentation
* **Classes**: `lower_left` (684 images)
* **Transformations**: None (online training-time standard normalizations only).

#### Tier 1: Very Light Augmentation
* **Classes**: `lower_occlusal` (621 images), `upper_occlusal` (560 images)
* **Parameters**:
  - Rotation: $\pm5^\circ$ (probability $p = 0.8$)
  - Brightness/Contrast: $\pm5\%$ ($p = 0.8$)

#### Tier 2: Moderate Augmentation
* **Classes**: `lower_front` (325 images), `lower_right` (278 images)
* **Parameters**:
  - Rotation: $\pm10^\circ$ ($p = 0.8$)
  - Brightness/Contrast: $\pm10\%$ ($p = 0.8$)
  - Crop Scale: $95\% - 100\%$ of original area ($p = 0.8$)
  - Gaussian Blur: $\sigma \in [0.5, 1.0]$ ($p = 0.5$)

#### Tier 3: Moderate-Aggressive Augmentation
* **Classes**: `upper_left` (198 images), `upper_front` (164 images)
* **Parameters**:
  - Rotation: $\pm12^\circ$ ($p = 0.8$)
  - Brightness/Contrast: $\pm12\%$ ($p = 0.8$)
  - Crop Scale: $90\% - 100\%$ of original area ($p = 0.8$)
  - Saturation shift: $\pm8\%$ ($p = 0.7$, Hue locked at 0)
  - Gaussian Blur: $\sigma \in [0.5, 1.5]$ ($p = 0.5$)

#### Tier 4: Aggressive Augmentation
* **Classes**: `upper_right` (145 images)
* **Parameters**:
  - Rotation: $\pm15^\circ$ ($p = 0.8$)
  - Brightness/Contrast: $\pm15\%$ ($p = 0.8$)
  - Crop Scale: $85\% - 100\%$ of original area ($p = 0.8$)
  - Saturation shift: $\pm10\%$ ($p = 0.7$, Hue locked at 0)
  - Gaussian Blur: $\sigma \in [0.5, 2.0]$ ($p = 0.5$)
  - Image Sharpening: $p = 0.5$

---

## 4. Final Balanced Dataset Cohort

Executing this tiered script yielded a highly balanced distribution across all eight clinical categories:

| Class ID | View Class | Unique Clean | Aug Created | Final Balanced | % of Dataset | Multiplier | Tier |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `lower_front` | 325 | 325 | 650 | 12.4% | 2.00x | Tier 2 |
| 1 | `lower_left` | 684 | 0 | 684 | 13.1% | 1.00x | Tier 0 |
| 2 | `lower_occlusal` | 621 | 29 | 650 | 12.4% | 1.05x | Tier 1 |
| 3 | `lower_right` | 278 | 372 | 650 | 12.4% | 2.34x | Tier 2 |
| 4 | `upper_front` | 164 | 486 | 650 | 12.4% | 3.96x | Tier 3 |
| 5 | `upper_left` | 198 | 452 | 650 | 12.4% | 3.28x | Tier 3 |
| 6 | `upper_occlusal` | 560 | 90 | 650 | 12.4% | 1.16x | Tier 1 |
| 7 | `upper_right` | 145 | 505 | 650 | 12.4% | 4.48x | Tier 4 |
| **Total** | **All Views** | **2,975** | **2,259** | **5,234** | **100%** | **1.76x** (Avg) | N/A |

### 4.1 Imbalance Reduction Metrics
- **Initial Dataset Imbalance Ratio**: $1129 / 146 =$ **$7.73\times$**
- **Cleaned Dataset Imbalance Ratio** (post-deduplication): $684 / 145 =$ **$4.72\times$**
- **Final Dataset Imbalance Ratio** (post-augmentation): $684 / 650 =$ **$1.05\times$**

The dataset is now effectively balanced. The remaining minimal discrepancy ($1.05\times$) will be handled seamlessly during the training phase using a `WeightedRandomSampler` or class weights in the PyTorch cross-entropy loss function.

---

## 5. Verification & Quality Assurance

To validate the clinical usability and dataset hygiene of the final augmented images, the automated `verify_augmentation.py` script ran a suite of integrity tests:

### 5.1 Test Suite Results
1. **File Integrity Scan**: Checked all 5,234 output files by attempting to open them with OpenCV. **0 corrupted files** were found; 100% of files are valid JPEGs.
2. **MD5 Duplication Audit**: Compared MD5 hashes across the entire 5,234 cohort. **0 duplicate hashes** were detected.
3. **Anatomical Correctness Audit**: Verified that no horizontal or vertical flips were included in any augmentation configuration.
4. **Visual Sample Grid Generation**: Selected four augmented classes (`lower_front`, `lower_occlusal`, `lower_right`, `upper_front`), extracted 1 original image and its 4 sequential augmented derivations, and compiled them into a sample comparison grid. The output is saved at `outputs/plots/augmentation_samples.png`.

### 5.2 Research Paper-Ready Methodology Paragraph
> *"To address severe class imbalance and remove data redundancy, we implemented a robust data engineering and duplicate-cleaning pipeline. Exact file copies were identified using MD5 cryptographic checksum hashing, revealing that 24.5% of the raw 3,950 images were redundant. Label noise was eliminated by completely removing four cross-class label conflicts. The remaining 2,975 unique images were standardized to a fixed input size of 224 x 224 pixels. A class-aware, tier-based offline augmentation pipeline was applied using the Albumentations library. Modalities of augmentation included small rotation ($\le15^\circ$), exposure jitter, localized random cropping, and Gaussian blurring to simulate clinical capture variations. Crucially, all flipping transforms (horizontal and vertical) were strictly prohibited to avoid left-right confusion and anatomical orientation corruption. An in-memory MD5 checksum validation filter guaranteed that all generated augmented images were unique. The final balanced dataset consists of 5,234 images with an imbalance ratio of 1.05x, ensuring highly stable gradient propagation and unbiased per-class view classification performance."*
