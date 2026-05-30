# Cross-Architecture Comparative Evaluation Summary

> **Task**: 8-Class Intraoral View Classification  
> **Dataset Split**: 15% Hold-out Test Set  
> **Date of Generation**: 2026-05-29 11:11:38  

## 1. Global Performance Metrics Table

The table below compares all evaluated architectures ordered by **Macro F1-Score** (primary metric for class-imbalance robustness).

| Model Architecture | Paradigm Family | Parameters | FLOPs | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Swin-Tiny | Vision Transformer | 28.3M | 4.5G | 0.9911 | 0.9913 | 0.9912 | 0.9911 |
| ConvNeXt-Tiny | Modern CNN | 28.6M | 4.5G | 0.9911 | 0.9911 | 0.9911 | 0.9911 |
| MobileNetV3-Small | Efficient CNN | 2.5M | 0.06G | 0.9898 | 0.9899 | 0.9900 | 0.9899 |
| DenseNet-121 | Classical CNN | 8.0M | 2.9G | 0.9898 | 0.9898 | 0.9898 | 0.9898 |
| ResNet-50 | Classical CNN | 25.6M | 4.1G | 0.9885 | 0.9886 | 0.9885 | 0.9885 |
| EfficientNet-B2 | Efficient CNN | 9.1M | 1.0G | 0.9835 | 0.9837 | 0.9835 | 0.9835 |
| EfficientNet-B3 | Efficient CNN | 12.2M | 1.8G | 0.9746 | 0.9748 | 0.9746 | 0.9746 |
| DINOv2-Small (ViT-S/14) | Vision Transformer | 22.1M | 4.6G | 0.9478 | 0.9521 | 0.9477 | 0.9481 |


## 2. Per-Class F1-Scores Breakdown

Detailed per-class F1-scores to identify performance disparities across dental views.

| Model | Lower Front | Lower Left | Lower Occlusal | Lower Right | Upper Front | Upper Left | Upper Occlusal | Upper Right |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Swin-Tiny | 0.9846 | 0.9902 | 1.0000 | 0.9949 | 0.9800 | 0.9897 | 1.0000 | 0.9897 |
| ConvNeXt-Tiny | 0.9949 | 0.9951 | 0.9949 | 1.0000 | 0.9796 | 0.9741 | 1.0000 | 0.9899 |
| MobileNetV3-Small | 0.9746 | 0.9852 | 1.0000 | 0.9848 | 0.9846 | 0.9949 | 1.0000 | 0.9949 |
| DenseNet-121 | 0.9948 | 1.0000 | 1.0000 | 0.9949 | 0.9645 | 0.9846 | 1.0000 | 0.9796 |
| ResNet-50 | 0.9948 | 1.0000 | 1.0000 | 0.9949 | 0.9697 | 0.9691 | 1.0000 | 0.9796 |
| EfficientNet-B2 | 0.9949 | 0.9902 | 0.9896 | 0.9898 | 0.9645 | 0.9846 | 1.0000 | 0.9548 |
| EfficientNet-B3 | 0.9796 | 0.9852 | 0.9845 | 0.9848 | 0.9637 | 0.9293 | 1.0000 | 0.9697 |
| DINOv2-Small (ViT-S/14) | 0.9697 | 0.9856 | 0.9949 | 0.9524 | 0.8651 | 0.9149 | 1.0000 | 0.9022 |
