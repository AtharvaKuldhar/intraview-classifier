# Dataset Analysis — Intraoral Dental Image Classification

> This document contains raw dataset statistics and augmentation details for the research paper.
> All numbers are verified via MD5 hash-based duplicate detection run on 2026-05-26.

---

## 1. Raw Dataset Overview

- **Source**: Intraoral dental photographs collected from clinical settings
- **Total raw images**: 3,950
- **Image format**: JPEG (.jpg)
- **Number of classes**: 8 (view-based classification)
- **Class naming convention**: `{jaw}_{view}` where jaw ∈ {upper, lower} and view ∈ {front, left, right, occlusal}

### 1.1 Raw Class Distribution

| Class | Raw Count | % of Total |
|---|---|---|
| Lower Left | 1,129 | 28.6% |
| Upper Occlusal | 987 | 25.0% |
| Lower Occlusal | 639 | 16.2% |
| Lower Front | 396 | 10.0% |
| Lower Right | 290 | 7.3% |
| Upper Left | 198 | 5.0% |
| Upper Front | 165 | 4.2% |
| Upper Right | 146 | 3.7% |
| **Total** | **3,950** | **100%** |

**Raw imbalance ratio** (max/min): 1,129 / 146 = **7.73x**

### 1.2 Image File Size Statistics (sampled)

| Class | Avg Size | Min Size | Max Size |
|---|---|---|---|
| Lower Front | 106 KB | 51 KB | 144 KB |
| Lower Left | 139 KB | 86 KB | 220 KB |
| Lower Occlusal | 129 KB | 118 KB | 136 KB |
| Lower Right | 335 KB | 109 KB | 1,019 KB |
| Upper Front | 123 KB | 87 KB | 148 KB |
| Upper Left | 66 KB | 48 KB | 101 KB |
| Upper Occlusal | 70 KB | 62 KB | 96 KB |
| Upper Right | 86 KB | 44 KB | 122 KB |

---

## 2. Duplicate Analysis

Duplicate detection was performed using MD5 file hashing across all 3,950 images.

### 2.1 Within-Class Duplicates

| Class | Raw Count | Unique Count | Duplicates Removed | Duplicate % |
|---|---|---|---|---|
| Lower Left | 1,129 | 686 | 443 | 39.2% |
| Upper Occlusal | 987 | 560 | 427 | 43.3% |
| Lower Occlusal | 639 | 621 | 18 | 2.8% |
| Lower Front | 396 | 329 | 67 | 16.9% |
| Lower Right | 290 | 279 | 11 | 3.8% |
| Upper Left | 198 | 198 | 0 | 0.0% |
| Upper Front | 165 | 165 | 0 | 0.0% |
| Upper Right | 146 | 145 | 1 | 0.7% |
| **Total** | **3,950** | **2,983** | **967** | **24.5%** |

**Key observation**: Nearly a quarter of the raw dataset consisted of exact duplicates. The two largest classes (Lower Left, Upper Occlusal) were the most affected, with 39–43% of their images being duplicates. This likely resulted from copy operations during initial data collection (evidenced by filenames containing "Copy" patterns).

### 2.2 Cross-Class Duplicates

4 images were identified as appearing in multiple class folders (identical MD5 hash, different class labels):

| Duplicate # | Classes Involved |
|---|---|
| 1 | Lower Front ↔ Lower Left |
| 2 | Lower Front ↔ Lower Right |
| 3 | Lower Front ↔ Lower Left |
| 4 | Lower Front ↔ Upper Front |

**Resolution**: All 4 cross-class duplicates (8 file copies total) were **removed from all classes**. An image that cannot be unambiguously assigned to a single view class represents label noise and would degrade classifier performance.

### 2.3 Clean Dataset (Post-Deduplication)

| Class | Clean Unique Count | % of Clean Total |
|---|---|---|
| Lower Left | 684 | 23.0% |
| Lower Occlusal | 621 | 20.9% |
| Upper Occlusal | 560 | 18.8% |
| Lower Front | 325 | 10.9% |
| Lower Right | 278 | 9.3% |
| Upper Left | 198 | 6.7% |
| Upper Front | 164 | 5.5% |
| Upper Right | 145 | 4.9% |
| **Total** | **2,975** | **100%** |

**Clean imbalance ratio** (max/min): 684 / 145 = **4.72x**

---

## 3. Data Augmentation Strategy

### 3.1 Design Principles

1. **No flips** — horizontal flips would transform left-view images into right-view appearances, contaminating class labels. Vertical flips are anatomically nonsensical.
2. **Medical realism** — all augmentations must produce clinically plausible images.
3. **Class-aware intensity** — augmentation aggressiveness scales inversely with class size.
4. **Maximum 5x multiplier** — to prevent excessive synthetic duplication per the project policy.
5. **Remaining imbalance handled at training time** via WeightedRandomSampler and weighted cross-entropy loss.

### 3.2 Augmentation Targets

Target per class: **~650 images** (classes already above this threshold are left as-is).

| Class | Clean Count | Target | Aug Images Needed | Multiplier | Tier |
|---|---|---|---|---|---|
| Lower Left | 684 | 684 (as-is) | 0 | 1.0x | Tier 0 |
| Lower Occlusal | 621 | 650 | 29 | 1.05x | Tier 1 |
| Upper Occlusal | 560 | 650 | 90 | 1.16x | Tier 1 |
| Lower Front | 325 | 650 | 325 | 2.0x | Tier 2 |
| Lower Right | 278 | 650 | 372 | 2.34x | Tier 2 |
| Upper Left | 198 | 650 | 452 | 3.28x | Tier 3 |
| Upper Front | 164 | 650 | 486 | 3.96x | Tier 3 |
| Upper Right | 145 | 650 | 505 | 4.48x | Tier 4 |

**Post-augmentation total: ~5,243 images**
**Post-augmentation imbalance ratio**: 684 / 645 = **1.06x** (effectively balanced)

### 3.3 Augmentation Transforms by Tier

Library: **albumentations**

All tiers use **only** the following safe transform families:
- Rotation (small angles)
- Brightness / Contrast adjustment
- Random resized crop
- Gaussian blur
- Saturation adjustment
- Sharpness adjustment

**Explicitly excluded transforms**: Horizontal flip, vertical flip, elastic distortion, grid distortion, CutOut, CutMix, random erasing, aggressive color jitter, channel shuffle.

#### Tier 0 — No Offline Augmentation
- Applied to: Lower Left (684 images)
- No offline augmentation; online training-time transforms only

#### Tier 1 — Very Light
- Applied to: Lower Occlusal (621 → 650), Upper Occlusal (560 → 650)
- Rotation: ±5°
- Brightness: ±5%
- Contrast: ±5%

#### Tier 2 — Moderate
- Applied to: Lower Front (325 → 650), Lower Right (278 → 650)
- Rotation: ±10°
- Brightness: ±10%
- Contrast: ±10%
- Gaussian blur: σ ∈ [0.5, 1.0]
- Random resized crop: scale ∈ [0.95, 1.0]

#### Tier 3 — Moderate-Aggressive
- Applied to: Upper Left (198 → 650), Upper Front (164 → 650)
- Rotation: ±12°
- Brightness: ±12%
- Contrast: ±12%
- Gaussian blur: σ ∈ [0.5, 1.5]
- Random resized crop: scale ∈ [0.90, 1.0]
- Saturation: ±8%

#### Tier 4 — Aggressive
- Applied to: Upper Right (145 → 650)
- Rotation: ±15°
- Brightness: ±15%
- Contrast: ±15%
- Gaussian blur: σ ∈ [0.5, 2.0]
- Random resized crop: scale ∈ [0.85, 1.0]
- Saturation: ±10%
- Sharpness adjustment

---

## 4. Summary Statistics for Paper

| Metric | Value |
|---|---|
| Raw images collected | 3,950 |
| Exact duplicates removed (within-class) | 967 (24.5%) |
| Cross-class label conflicts removed | 8 files (4 unique images) |
| Clean unique images | 2,975 |
| Number of classes | 8 |
| Raw imbalance ratio | 7.73x |
| Clean imbalance ratio | 4.72x |
| Post-augmentation images | ~5,243 |
| Post-augmentation imbalance ratio | ~1.06x |
| Augmentation library | albumentations |
| Max augmentation multiplier used | 4.48x (Upper Right) |
| Augmentation families | Rotation, brightness, contrast, blur, crop, saturation, sharpness |
| Augmentations excluded | All flips, elastic/grid distortion, CutOut, CutMix, random erasing |
