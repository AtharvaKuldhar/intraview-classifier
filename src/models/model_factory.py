import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Tuple

# Comprehensive mapping of architectures to their exact PyTorch Image Models (timm) identifiers.
# These have been verified as paper-ready and available in modern timm versions.
MODEL_REGISTRY = {
    "resnet50": "resnet50.a1_in1k",
    "densenet121": "densenet121.ra_in1k",
    "mobilenetv3_small": "mobilenetv3_small_100.lamb_in1k",
    "efficientnet_b2": "efficientnet_b2.ra_in1k",
    "efficientnet_b3": "efficientnet_b3.ra2_in1k",
    "convnext_tiny": "convnext_tiny.in12k_ft_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "dinov2_small": "vit_small_patch14_dinov2.lvd142m"
}

def create_model(model_name: str, pretrained: bool = True, num_classes: int = 8) -> nn.Module:
    """
    Unified Strategy Pattern for Model Architecture Instantiation.
    Leverages PyTorch Image Models (timm) for standard backbones and head adaptations.
    
    Args:
        model_name (str): Key name of the model from MODEL_REGISTRY.
        pretrained (bool): If True, loads pre-trained ImageNet weights.
        num_classes (int): Number of final target view classes. Defaults to 8.
        
    Returns:
        nn.Module: The constructed PyTorch model with custom classification head.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            f"Please choose from: {list(MODEL_REGISTRY.keys())}"
        )
        
    timm_name = MODEL_REGISTRY[model_name]
    print(f"[MODEL_FACTORY] Instantiating '{model_name}' using timm backbone: '{timm_name}'")
    
    # 1. Instantiate the pre-trained architecture with adapted head size
    try:
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create model {model_name} ({timm_name}) via timm: {e}. "
            "Verify internet connection and timm library version."
        )
        
    # Log model statistics
    total_params, trainable_params = get_parameter_count(model)
    print(f"[MODEL_FACTORY] Success! Total Params: {total_params:,} | Trainable: {trainable_params:,}")
    
    return model

def freeze_backbone(model: nn.Module) -> None:
    """
    Freezes all encoder layers (backbone) of the model, leaving only
    the final classification head active. Used in Phase 1 (feature extraction) training.
    
    Args:
        model (nn.Module): PyTorch model to freeze.
    """
    # 1. Freeze every parameter in the network first
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Safely unfreeze the classification head using timm's classifier utility
    try:
        classifier = model.get_classifier()
        for param in classifier.parameters():
            param.requires_grad = True
        print("[MODEL_FACTORY] Backbone frozen. Switched to Head-Only Training (Phase 1).")
    except Exception as e:
        # Fallback: Find standard head naming schemes in PyTorch/timm if get_classifier fails
        unfrozen = False
        for head_name in ['fc', 'classifier', 'head', 'logits']:
            if hasattr(model, head_name):
                head = getattr(model, head_name)
                for param in head.parameters():
                    param.requires_grad = True
                unfrozen = True
                break
        if unfrozen:
            print("[MODEL_FACTORY] Backbone frozen (Fallback head detection). Switched to Head-Only Training.")
        else:
            raise RuntimeError(f"Could not identify classification head to unfreeze: {e}")

def unfreeze_all_parameters(model: nn.Module) -> None:
    """
    Unfreezes all parameters across all layers in the network.
    Used in Phase 2 (full fine-tuning) training.
    
    Args:
        model (nn.Module): PyTorch model to fully unfreeze.
    """
    for param in model.parameters():
        param.requires_grad = True
    print("[MODEL_FACTORY] Unfroze all layers. Switched to Full Model Fine-Tuning (Phase 2).")

def get_parameter_count(model: nn.Module) -> Tuple[int, int]:
    """
    Computes total parameter count and currently trainable parameter count.
    
    Args:
        model (nn.Module): The model to analyze.
        
    Returns:
        Tuple[int, int]: (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
