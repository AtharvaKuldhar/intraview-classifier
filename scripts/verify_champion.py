"""
Champion Model Verification Script
===================================
Run this in Google Colab to identify the actual champion model from all trained metrics.
Prints a ranked leaderboard of all models with their test performance.

Usage:
    !python scripts/verify_champion.py
"""
import os
import sys
import json

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_META = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "mobilenetv3_small": "MobileNetV3-Small",
    "efficientnet_b2": "EfficientNet-B2",
    "efficientnet_b3": "EfficientNet-B3",
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_tiny": "Swin-Tiny",
    "dinov2_small": "DINOv2-Small (ViT-S/14)"
}

def main():
    is_colab = os.path.exists("/content")
    drive_mounted = os.path.exists("/content/drive")
    
    search_roots = []
    if is_colab and drive_mounted:
        search_roots.append("/content/drive/MyDrive/dental_research")
    search_roots.extend(["checkpoints", "experiments"])
    
    print("=" * 70)
    print("CHAMPION MODEL VERIFICATION — RANKED LEADERBOARD")
    print("=" * 70)
    
    # Collect all metrics.json
    results = []
    seen = set()
    
    for root_path in search_roots:
        if not os.path.exists(root_path):
            continue
        for dirpath, dirnames, filenames in os.walk(root_path):
            if "metrics.json" in filenames:
                metrics_path = os.path.join(dirpath, "metrics.json")
                abs_path = os.path.abspath(metrics_path)
                if abs_path in seen:
                    continue
                seen.add(abs_path)
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    model_name = data.get("model_name", "")
                    if model_name in MODEL_META:
                        gm = data.get("global_metrics", {})
                        results.append({
                            "model_name": model_name,
                            "display_name": MODEL_META[model_name],
                            "accuracy": gm.get("Accuracy", 0.0),
                            "precision": gm.get("Precision_Macro", 0.0),
                            "recall": gm.get("Recall_Macro", 0.0),
                            "f1_macro": gm.get("F1_Macro", 0.0),
                            "epoch": data.get("checkpoint_epoch", "?"),
                            "source": metrics_path
                        })
                        print(f"  [OK] Loaded: {MODEL_META[model_name]} from {metrics_path}")
                except Exception as e:
                    print(f"  [WARN] Error reading {metrics_path}: {e}")
                    
    if not results:
        print("\n[ERROR] No metrics.json files found. Run evaluator.py on all models first!")
        return
        
    # Sort by F1 Macro descending
    results.sort(key=lambda x: x["f1_macro"], reverse=True)
    
    print(f"\n{'='*70}")
    print(f"{'Rank':<6} {'Model':<28} {'Accuracy':>10} {'Precision':>11} {'Recall':>10} {'F1 Macro':>10}")
    print(f"{'-'*70}")
    
    for rank, r in enumerate(results, 1):
        marker = " ★ CHAMPION" if rank == 1 else ""
        print(f"  {rank:<4} {r['display_name']:<28} {r['accuracy']*100:>8.2f}%  {r['precision']*100:>8.2f}%  {r['recall']*100:>8.2f}%  {r['f1_macro']*100:>8.2f}%{marker}")
        
    print(f"{'='*70}")
    
    champion = results[0]
    print(f"\n🏆 VERIFIED CHAMPION: {champion['display_name']}")
    print(f"   Macro F1-Score: {champion['f1_macro']*100:.2f}%")
    print(f"   Test Accuracy:  {champion['accuracy']*100:.2f}%")
    print(f"   Best Epoch:     {champion['epoch']}")
    print(f"   Metrics File:   {champion['source']}")
    
    if len(results) >= 2:
        runner_up = results[1]
        margin = (champion["f1_macro"] - runner_up["f1_macro"]) * 100
        print(f"\n   Lead over runner-up ({runner_up['display_name']}): +{margin:.2f}% F1 Macro")
        
    print(f"\n{'='*70}")
    print(f"Use the champion config for K-Fold Cross Validation:")
    print(f"  !python scripts/run_kfold.py --config src/configs/{champion['model_name']}.yaml --k 5")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
