# Lesson 04 — Vision Transformer Positional Embeddings & Dynamic Grid Interpolation

> **Role**: Explain Agent (ML Framework Architect + Educator)  
> **Topic**: Vision Transformers, Positional Embeddings, Patch Embedding Assertions, and Bicubic Interpolation in PyTorch Image Models (timm)  
> **Target Audience**: Dental Image Classifier Research Paper Reference

This document logs the **what**, **why**, **how**, and academic **tradeoffs** of Vision Transformer spatial resolution adaptation, specifically addressing the DINOv2 `AssertionError` (Input Height 224 vs Model Native Height 518).

---

## 1. The Core Issue (The "What")

When instantiating DINOv2-Small (`vit_small_patch14_dinov2.lvd142m`) and training it on $224 \times 224$ dental images, the training process crashed during the first forward pass with the following assertion:
```text
File "/usr/local/lib/python3.12/dist-packages/timm/layers/patch_embed.py", line 121, in forward
  _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
AssertionError: Input height (224) doesn't match model (518).
```

### Why Classical CNNs (ResNet) DO NOT have this problem:
* Classical convolutional layers process images using sliding filters that apply identical sliding multiplication operations regardless of image size. 
* A $3 \times 3$ convolution filter can process a $224 \times 224$ image or a $518 \times 518$ image identically—it simply outputs a larger or smaller feature map. CNNs are **spatially invariant**.

### Why Vision Transformers (ViTs) DO have this problem:
* Vision Transformers do not use convolutions. Instead, they flatten an image into a 1D sequence of visual tokens (patches) and process them using standard self-attention blocks.
* Because self-attention is permutation-invariant (it does not know the relative spatial ordering of patches), ViTs **must inject positional embeddings** into the patch sequences so the network retains coordinates.
* These positional embeddings are **static parameters** linked to a specific grid resolution. If the input resolution does not match the grid, the number of tokens is incorrect, and the model crashes.

---

## 2. Mathematical Breakdown (The "Why")

Let's trace the math of the Patch Embedding layer for DINOv2:

```
                  ┌───────────────────────────────┐
                  │      Original Input Image     │
                  │   Height (H) x Width (W)      │
                  └───────────────┬───────────────┘
                                  │
                                  v
                  ┌───────────────────────────────┐
                  │    Patch Partition Block      │
                  │     Patch Size (P) = 14       │
                  └───────────────┬───────────────┘
                                  │
                                  v
                  ┌───────────────────────────────┐
                  │      Visual Token Sequence    │
                  │  N = (H / P) x (W / P) tokens │
                  └───────────────────────────────┘
```

### Case A: The Pre-trained Native Setup ($518 \times 518$)
1. DINOv2 was pre-trained on Meta's LVD-142M dataset using images resized to $518 \times 518$.
2. The patch size is $P = 14$ pixels.
3. The number of spatial patches is:
   $$\text{Grid} = \left(\frac{518}{14}\right) \times \left(\frac{518}{14}\right) = 37 \times 37 = 1,369\text{ patches}$$
4. The model weight dictionary contains a pre-trained positional embedding parameter of shape `[1, 1370, 384]` (1,369 spatial tokens + 1 extra `[CLS]` token, with a feature dimension of 384).

### Case B: Our Dental Classifier Setup ($224 \times 224$)
1. Our balanced dataset is standardized to $224 \times 224$ pixels.
2. The patch size remains $P = 14$ pixels.
3. The number of spatial patches is:
   $$\text{Grid} = \left(\frac{224}{14}\right) \times \left(\frac{224}{14}\right) = 16 \times 16 = 256\text{ patches}$$
4. The model expects to add a positional embedding of shape `[1, 257, 384]`.
5. **The Conflict**: PyTorch tries to add the native `[1, 1370, 384]` positional embedding to the incoming `[1, 257, 384]` sequence. This shape mismatch causes an immediate mathematical crash!

---

## 3. How We Solved It: Bicubic Grid Interpolation

To solve this resolution conflict, we must dynamically resize the $37 \times 37$ grid of pre-trained positional embeddings down to a $16 \times 16$ grid. 

In `timm`, we accomplish this by passing the target resolution `img_size` directly to the model constructor inside [`model_factory.py`](file:///d:/Dental_Image_Classifier/src/models/model_factory.py):

```python
# Check if the model requires spatial configuration (ViTs/Swin)
kwargs = {}
if "dinov2" in model_name or "vit" in model_name or "swin" in model_name:
    kwargs["img_size"] = img_size  # Explicitly sets target img_size (e.g., 224)
    
model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes, **kwargs)
```

### What happens under the hood during this call?
When `timm.create_model` detects that `img_size=224` does not match the checkpoint's native resolution ($518$), it automatically triggers an **adaptive resampling** routine:
1. **Extraction**: It strips the `[CLS]` token embedding from the positional embedding tensor.
2. **Reshaping**: It reshapes the remaining $1,369$ spatial coordinates from a 1D sequence of length $1369$ back to their 2D grid shape of `[1, 384, 37, 37]`.
3. **Resampling**: It applies **Bicubic Interpolation** (resizing) to scale the $37 \times 37$ spatial matrix down to our target $16 \times 16$ size, resulting in a shape of `[1, 384, 16, 16]`.
4. **Assembly**: It flattens the resampled $16 \times 16$ matrix back to $256$ spatial coordinates, prepends the `[CLS]` token back, and updates the model's parameters to `[1, 257, 384]`.
5. **Re-configuration**: It updates the model's internally asserted `PatchEmbed.img_size` to `(224, 224)`.

---

## 4. Tradeoffs & Alternatives

### Tradeoff: Positional Grid Distortions
* **The Tradeoff**: Rescaling positional embeddings from $37 \times 37$ down to $16 \times 16$ means the spatial coordinates are slightly compressed. The spatial frequency learned during pre-training is altered. 
* **The Reality**: Extensive research shows that **bicubic interpolation** preserves spatial continuity incredibly well. The rescaled model retains a strong sense of spatial coordinates, and the performance penalty is virtually negligible compared to the benefits of using pre-trained features.

### Alternative A: Training at Native $518 \times 518$ Resolution
* **What**: Scale our dataset up to $518 \times 518$ to match DINOv2 natively.
* **Why we rejected it**: Spatially scaling images up by $2.3\text{x}$ increases the number of pixels per image by **$5.3\text{x}$** ($518^2 \approx 268,000$ pixels vs $224^2 \approx 50,000$ pixels). This would:
  1. Increase training time on Google Colab T4 GPU by over **$5\text{x}$**.
  2. Cause Out-Of-Memory (OOM) VRAM errors on Colab T4 GPU unless the batch size was dropped to $4$ or $8$, which would destabilize gradients.
  3. Introduce synthetic interpolation artifacts into your raw images.

### Alternative B: Deactivating Positional Embeddings
* **What**: Stop injecting positional coordinates into the sequence.
* **Why we rejected it**: Disabling positional coordinates reduces the Vision Transformer to a "bag of patches." The model would lose all concept of structural arrangement—it wouldn't know if a tooth was on the left or the right side of the arch, or if it was in the upper or lower jaw. This destroys view classification accuracy.
