# pyrefly: ignore [missing-import]
import albumentations as A
from typing import Optional, Dict, Any

def get_transforms_for_tier(tier: int, tier_config: Dict[str, Any]) -> Optional[A.Compose]:
    """
    Returns an Albumentations composition for the specified tier.
    
    CRITICAL POLICY RESTRICTION (as per AGENTS.md):
    - NO FLIPS allowed (horizontal or vertical). Horizontal flips swap left/right classes,
      and vertical flips violate anatomical orientation.
    - NO extreme deformations (elastic transforms, grid distortion). They create tooth 
      structures that are physically impossible in real clinical cases.
    - NO cutout/erasing. They can obscure critical diagnostic tooth and gum structures.
    """
    if tier == 0:
        # Tier 0 has no offline augmentation
        return None
        
    transforms_list = []
    
    # 1. Spatial Transforms: Rotation & Cropping (No Flips!)
    if "rotation_limit" in tier_config:
        limit = tier_config["rotation_limit"]
        # Rotate: Small rotation simulates slight patient head tilt or camera angle variation.
        # border_mode=0 (constant black border) or border_mode=4 (reflect) is standard;
        # in dental photos, reflect or replicate is best to avoid sharp black artificial borders.
        transforms_list.append(A.Rotate(limit=limit, p=0.8, border_mode=4))
        
    if "crop_scale_min" in tier_config and "crop_scale_max" in tier_config:
        scale_min = tier_config["crop_scale_min"]
        scale_max = tier_config["crop_scale_max"]
        # RandomResizedCrop: Simulates camera zoom variations or cropping differences 
        # while keeping the central aspect ratio. Scale controls how much zoom is allowed.
        # Standard clinical views focus on a specific arch section, so we keep zoom mild (e.g. >= 85%).
        transforms_list.append(A.RandomResizedCrop(
            size=(224, 224), # We resize to a standard CNN size, say 224x224 (adjustable)
            scale=(scale_min, scale_max),
            ratio=(0.95, 1.05),
            p=0.8
        ))
    else:
        # If no cropping, we just resize to standard dimensions
        transforms_list.append(A.Resize(height=224, width=224))

    # 2. Color / Contrast Transforms: Simulates different camera flashes, exposures, and sensors.
    if "brightness_limit" in tier_config or "contrast_limit" in tier_config:
        bright = tier_config.get("brightness_limit", 0.0)
        contrast = tier_config.get("contrast_limit", 0.0)
        # RandomBrightnessContrast: Essential for dental photos because different intraoral cameras
        # and lightning conditions (ambient overhead light vs Ring Flash) produce varying exposures.
        transforms_list.append(A.RandomBrightnessContrast(
            brightness_limit=bright,
            contrast_limit=contrast,
            p=0.8
        ))
        
    if "saturation_limit" in tier_config:
        sat = tier_config["saturation_limit"]
        # HueSaturationValue: Adjusts tissue coloration.
        # We LOCK hue_shift_limit to 0 because changing the hue of gums (pink) or teeth (white/yellow)
        # would look medically unrealistic (e.g., green gums). Saturation is altered slightly to
        # simulate different camera sensor color richness.
        transforms_list.append(A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=sat,
            val_shift_limit=0,
            p=0.7
        ))

    # 3. Focus / Noise Transforms: Simulates lens focus issues or sensor noise
    if "blur_sigma_min" in tier_config and "blur_sigma_max" in tier_config:
        sigma_min = tier_config["blur_sigma_min"]
        sigma_max = tier_config["blur_sigma_max"]
        # GaussianBlur: Simulates slight defocusing of the lens, which is extremely common when 
        # taking quick pictures inside a patient's mouth.
        transforms_list.append(A.GaussianBlur(
            blur_limit=(3, 7), # Standard odd kernel sizes
            sigma_limit=(sigma_min, sigma_max),
            p=0.5
        ))
        
    # 4. Detail / Sharpness Transforms (Tier 4 only)
    if tier_config.get("sharpen", False):
        # Sharpen: Accentuate details (e.g., enamel boundaries, crack lines).
        # Sometimes helps high-end sensors' sharp details look matching.
        transforms_list.append(A.Sharpen(p=0.5))
        
    return A.Compose(transforms_list)
