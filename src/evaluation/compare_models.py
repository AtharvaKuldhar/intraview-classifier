import os
import sys
import json
import argparse
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.datasets.dental_dataset import DentalDataset

# Static architectural metadata matching model_specifications.md for research integrity
MODEL_META = {
    "resnet50": {"display_name": "ResNet-50", "params": "25.6M", "flops": "4.1G", "family": "Classical CNN"},
    "densenet121": {"display_name": "DenseNet-121", "params": "8.0M", "flops": "2.9G", "family": "Classical CNN"},
    "mobilenetv3_small": {"display_name": "MobileNetV3-Small", "params": "2.5M", "flops": "0.06G", "family": "Efficient CNN"},
    "efficientnet_b2": {"display_name": "EfficientNet-B2", "params": "9.1M", "flops": "1.0G", "family": "Efficient CNN"},
    "efficientnet_b3": {"display_name": "EfficientNet-B3", "params": "12.2M", "flops": "1.8G", "family": "Efficient CNN"},
    "convnext_tiny": {"display_name": "ConvNeXt-Tiny", "params": "28.6M", "flops": "4.5G", "family": "Modern CNN"},
    "swin_tiny": {"display_name": "Swin-Tiny", "params": "28.3M", "flops": "4.5G", "family": "Vision Transformer"},
    "dinov2_small": {"display_name": "DINOv2-Small (ViT-S/14)", "params": "22.1M", "flops": "4.6G", "family": "Vision Transformer"}
}

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model Performance Comparison and Report Generator.")
    parser.add_argument(
        "--experiments_dir", 
        type=str, 
        default="experiments", 
        help="Root folder containing the individual model training runs."
    )
    parser.add_argument(
        "--drive_prefix", 
        type=str, 
        default=None, 
        help="Optional drive path for Google Colab run redirection."
    )
    return parser.parse_args()

def collect_experiment_metrics(experiments_root: str) -> List[Dict[str, Any]]:
    """Recursively walks the search directories to gather all metrics.json files."""
    results = []
    
    search_paths = []
    if experiments_root:
        search_paths.append(experiments_root)
        
        # If the provided path ends with "experiments", also search the sibling "checkpoints" directory
        norm_path = experiments_root.rstrip("/\\")
        if os.path.basename(norm_path) == "experiments":
            parent = os.path.dirname(norm_path)
            checkpoints_sibling = os.path.join(parent, "checkpoints")
            if os.path.exists(checkpoints_sibling):
                search_paths.append(checkpoints_sibling)
                print(f"[COMPARE] Sibling checkpoints directory detected: adding '{checkpoints_sibling}' to crawl path.")
        else:
            # If the provided path itself is the root dental_research directory, look for nested checkpoints
            checkpoints_nested = os.path.join(norm_path, "checkpoints")
            if os.path.exists(checkpoints_nested) and checkpoints_nested not in search_paths:
                search_paths.append(checkpoints_nested)
                print(f"[COMPARE] Nested checkpoints directory detected: adding '{checkpoints_nested}' to crawl path.")
                
    # Also search local folders if they exist
    for fallback in ["checkpoints", "experiments"]:
        if os.path.exists(fallback) and os.path.abspath(fallback) not in [os.path.abspath(p) for p in search_paths]:
            search_paths.append(fallback)
            
    print(f"[COMPARE] Crawling directories recursively for metrics: {search_paths}")
    
    seen_metrics = set()
    for path in search_paths:
        if not os.path.exists(path):
            continue
        for root, dirs, files in os.walk(path):
            if "metrics.json" in files:
                metrics_json = os.path.join(root, "metrics.json")
                abs_metrics_json = os.path.abspath(metrics_json)
                if abs_metrics_json in seen_metrics:
                    continue
                seen_metrics.add(abs_metrics_json)
                try:
                    with open(metrics_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    model_name = data.get("model_name")
                    if model_name in MODEL_META:
                        meta = MODEL_META[model_name]
                        folder_name = os.path.basename(root)
                        record = {
                            "model_name": model_name,
                            "display_name": meta["display_name"],
                            "family": meta["family"],
                            "params": meta["params"],
                            "flops": meta["flops"],
                            "epoch": data.get("checkpoint_epoch"),
                            "experiment_folder": folder_name
                        }
                        # Flatten global metrics
                        global_metrics = data.get("global_metrics", {})
                        record.update(global_metrics)
                        
                        # Store per-class F1 scores directly for plotting
                        class_perf = data.get("class_performance", {})
                        for c in DentalDataset.CLASSES:
                            record[f"F1_{c}"] = class_perf.get(c, {}).get("F1", 0.0)
                            
                        results.append(record)
                        print(f"[COMPARE] Successfully collected metrics for '{model_name}' from: {metrics_json}")
                except Exception as e:
                    print(f"[COMPARE] Error reading {metrics_json}: {e}")
                    
    return results

def generate_markdown_report(df: pd.DataFrame, save_path: str):
    """Generates a detailed markdown report with comparison tables."""
    # Order by F1 Macro descending
    df_sorted = df.sort_values(by="F1_Macro", ascending=False)
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("# Cross-Architecture Comparative Evaluation Summary\n\n")
        f.write("> **Task**: 8-Class Intraoral View Classification  \n")
        f.write("> **Dataset Split**: 15% Hold-out Test Set  \n")
        f.write(f"> **Date of Generation**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
        
        f.write("## 1. Global Performance Metrics Table\n\n")
        f.write("The table below compares all evaluated architectures ordered by **Macro F1-Score** (primary metric for class-imbalance robustness).\n\n")
        
        cols = ["display_name", "family", "params", "flops", "Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro"]
        headers = ["Model Architecture", "Paradigm Family", "Parameters", "FLOPs", "Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]
        
        # Write headers
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        
        for _, row in df_sorted.iterrows():
            row_vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    row_vals.append(f"{val:.4f}")
                else:
                    row_vals.append(str(val))
            f.write("| " + " | ".join(row_vals) + " |\n")
            
        f.write("\n\n## 2. Per-Class F1-Scores Breakdown\n\n")
        f.write("Detailed per-class F1-scores to identify performance disparities across dental views.\n\n")
        
        class_cols = ["display_name"] + [f"F1_{c}" for c in DentalDataset.CLASSES]
        class_headers = ["Model"] + [c.replace("_", " ").title() for c in DentalDataset.CLASSES]
        
        f.write("| " + " | ".join(class_headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(class_headers)) + " |\n")
        
        for _, row in df_sorted.iterrows():
            row_vals = []
            for col in class_cols:
                val = row[col]
                if isinstance(val, float):
                    row_vals.append(f"{val:.4f}")
                else:
                    row_vals.append(str(val))
            f.write("| " + " | ".join(row_vals) + " |\n")

def generate_latex_table(df: pd.DataFrame, save_path: str):
    """Generates a publication-ready LaTeX table block for the research paper."""
    df_sorted = df.sort_values(by="F1_Macro", ascending=False)
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("% ==========================================================================\n")
        f.write("% LaTeX Table Code block for Section 5 (Results) of the Dental Classifier Paper\n")
        f.write("% Generated automatically by compare_models.py\n")
        f.write("% ==========================================================================\n\n")
        
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparative performance analysis of CNN and Vision Transformer architectures on the 8-class intraoral view classification dataset (hold-out test split). Models are ordered by Macro $F_1$-Score.}\n")
        f.write("\\label{tab:model_comparison}\n")
        f.write("\\begin{tabular}{llccccccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Model Architecture} & \\textbf{Paradigm Family} & \\textbf{Params} & \\textbf{FLOPs} & \\textbf{Accuracy} & \\textbf{Precision}_{\\text{Macro}} & \\textbf{Recall}_{\\text{Macro}} & \\textbf{F}_1\\textbf{-Score}_{\\text{Macro}} \\\\\n")
        f.write("\\hline\n")
        
        for _, row in df_sorted.iterrows():
            name = row["display_name"]
            family = row["family"]
            params = row["params"]
            flops = row["flops"]
            acc = row["Accuracy"]
            prec = row["Precision_Macro"]
            rec = row["Recall_Macro"]
            f1 = row["F1_Macro"]
            
            # Format numbers to LaTeX syntax
            f.write(f"{name} & {family} & {params} & {flops} & {acc:.4f} & {prec:.4f} & {rec:.4f} & \\mathbf{{{f1:.4f}}} \\\\\n" if row.name == df_sorted.index[0] 
                    else f"{name} & {family} & {params} & {flops} & {acc:.4f} & {prec:.4f} & {rec:.4f} & {f1:.4f} \\\\\n")
            
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

def plot_grouped_comparison(df: pd.DataFrame, save_path: str):
    """Plots a high-quality grouped bar chart comparing global metrics across models."""
    df_sorted = df.sort_values(by="F1_Macro", ascending=True)  # Ascending for horizontal layout
    
    models = df_sorted["display_name"].tolist()
    accuracies = df_sorted["Accuracy"].tolist()
    f1_macros = df_sorted["F1_Macro"].tolist()
    
    y = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    rects1 = ax.barh(y - width/2, f1_macros, width, label='Macro F1-Score', color='#1f77b4')
    rects2 = ax.barh(y + width/2, accuracies, width, label='Accuracy', color='#ff7f0e')
    
    ax.set_ylabel('Model Architecture', fontsize=12, fontweight='bold')
    ax.set_xlabel('Performance Score', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Architecture Global Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=10, fontweight='semibold')
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=11)
    
    # Annotate bars with values
    def autolabel(rects):
        for rect in rects:
            width_val = rect.get_width()
            ax.annotate(f'{width_val:.3f}',
                        xy=(width_val, rect.get_y() + rect.get_height() / 2),
                        xytext=(3, 0),  # 3 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center', fontsize=8, fontweight='bold')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_class_f1_heatmap(df: pd.DataFrame, save_path: str):
    """Plots a comparative per-class heatmap for all models."""
    # Set model name as index
    df_indexed = df.set_index("display_name")
    
    # Extract only class F1 columns
    class_cols = [f"F1_{c}" for c in DentalDataset.CLASSES]
    f1_df = df_indexed[class_cols]
    
    # Rename columns for display
    f1_df.columns = [c.replace("F1_", "").replace("_", " ").title() for c in f1_df.columns]
    
    # Sort index descending by average F1
    f1_df = f1_df.loc[df.sort_values(by="F1_Macro", ascending=False)["display_name"]]
    
    plt.figure(figsize=(12, 6))
    
    # Draw custom heatmap without seaborn dependency
    im = plt.imshow(f1_df.values, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
    plt.colorbar(im, label='F1-Score')
    
    plt.title('Per-Class F1-Score Architectural Performance Disparities', fontsize=14, fontweight='bold', pad=20)
    
    # Set ticks
    plt.xticks(np.arange(len(f1_df.columns)), f1_df.columns, rotation=45, ha='right', fontsize=10, fontweight='semibold')
    plt.yticks(np.arange(len(f1_df.index)), f1_df.index, fontsize=10, fontweight='semibold')
    
    # Write values inside cells
    for i in range(f1_df.shape[0]):
        for j in range(f1_df.shape[1]):
            val = f1_df.values[i, j]
            plt.text(
                j, i, f"{val:.3f}",
                ha="center", va="center", 
                color="black" if 0.7 < val < 0.95 else "white" if val >= 0.95 or val <= 0.7 else "black",
                fontweight='bold', fontsize=9
            )
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    is_colab = os.path.exists("/content")
    drive_mounted = os.path.exists("/content/drive")
    
    # Resolve output root and drive prefix redirection dynamically
    output_root = ""
    if args.drive_prefix and args.drive_prefix != "True":
        output_root = args.drive_prefix
    elif is_colab and drive_mounted:
        output_root = "/content/drive/MyDrive/dental_research"
        
    # Setup search path
    experiments_root = args.experiments_dir
    
    # If output_root is resolved, align experiments_root to the output_root if it was not customized
    if output_root and (experiments_root == "experiments" or not os.path.isabs(experiments_root)):
        experiments_root = os.path.join(output_root, "experiments")
        
    print(f"[COMPARE] Gathering results from folder: '{experiments_root}'")
    
    # 1. Gather all experiment metrics JSON files
    results_list = collect_experiment_metrics(experiments_root)
    
    if len(results_list) == 0:
        print("[COMPARE] Critical: No metrics.json files were found. Make sure models have been trained and evaluated first!")
        return
        
    df = pd.DataFrame(results_list)
    
    # Setup output paths for plots and reports
    global_reports_dir = os.path.join(output_root, "outputs", "reports") if output_root else "outputs/reports"
    global_plots_dir = os.path.join(output_root, "outputs", "plots") if output_root else "outputs/plots"
    
    os.makedirs(global_reports_dir, exist_ok=True)
    os.makedirs(global_plots_dir, exist_ok=True)
    
    # 2. Generate Reports
    md_save_path = os.path.join(global_reports_dir, "model_comparison.md")
    latex_save_path = os.path.join(global_reports_dir, "model_comparison.tex")
    
    generate_markdown_report(df, md_save_path)
    print(f"[COMPARE] Generated Markdown report at: {md_save_path}")
    
    generate_latex_table(df, latex_save_path)
    print(f"[COMPARE] Generated publication-ready LaTeX table block at: {latex_save_path}")
    
    # 3. Generate Comparative Plots
    bar_chart_save_path = os.path.join(global_plots_dir, "model_comparison_bar.png")
    plot_grouped_comparison(df, bar_chart_save_path)
    print(f"[COMPARE] Generated grouped performance chart at: {bar_chart_save_path}")
    
    heatmap_save_path = os.path.join(global_plots_dir, "per_class_f1_comparison.png")
    plot_per_class_f1_heatmap(df, heatmap_save_path)
    print(f"[COMPARE] Generated per-class performance heatmap at: {heatmap_save_path}")
    
    print("\n" + "=" * 50)
    print("SUCCESS: Performance comparative figures and paper assets compiled successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
