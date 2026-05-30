"""
K-Fold Cross-Validation Inference Plotter
========================================
Generates publication-quality, high-DPI scientific visualizations from K-Fold cross-validation report JSON files.

Visualizations generated:
1. Performance Boxplot (Distribution of F1-Scores across different K)
2. Data Scaling Curve (Mean Performance with Standard Deviation Shaded Band)
3. Multi-Metric Convergence (Line Comparison of Acc, Prec, Rec, F1 across K)
4. Fold-Level Beeswarm/Scatter Plot (Individual fold scores to highlight clinical outliers)

Usage:
    python scripts/plot_kfold_inference.py --reports_dir outputs/kfold --output_dir outputs/plots/kfold
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Set modern scientific aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Inter', 'Roboto'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Harmonious research color palette (HSL tailored colors)
COLORS = {
    'primary': '#2A52BE',     # Classic Blue
    'secondary': '#E06666',   # Soft Red/Coral
    'accent': '#6AA84F',      # Clinical Green
    'warning': '#F1C232',     # Amber
    'neutral_dark': '#2B2B2B',# Charcoal
    'band_fill': '#D1E8FF'     # Light Blue for Std Shaded Band
}

def parse_args():
    parser = argparse.ArgumentParser(description="Publication-ready K-Fold Inference Visualization Engine.")
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="outputs/kfold",
        help="Path to directory containing K-Fold JSON report files (e.g., swin_tiny_k2_report.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/plots/kfold",
        help="Path to directory where generated plots will be saved."
    )
    return parser.parse_args()

def load_kfold_data(reports_dir: str) -> List[Dict[str, Any]]:
    """Loads all swin_tiny K-Fold reports found in the reports directory."""
    k_reports = []
    
    if not os.path.exists(reports_dir):
        print(f"[ERROR] Reports directory '{reports_dir}' does not exist.")
        print("[TIP] Make sure to download your JSON report files from Google Drive and place them in 'outputs/kfold/'.")
        return []
        
    for filename in sorted(os.listdir(reports_dir)):
        if filename.startswith("swin_tiny_k") and filename.endswith("_report.json"):
            filepath = os.path.join(reports_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                k_reports.append(data)
                print(f"[LOAD] Loaded report: '{filename}' (K={data['k']})")
            except Exception as e:
                print(f"[WARN] Error reading {filename}: {e}")
                
    # Sort by K value
    k_reports.sort(key=lambda x: x["k"])
    return k_reports

def plot_boxplots(reports: List[Dict[str, Any]], output_path: str):
    """Generates a boxplot showing F1-Score distributions across different K values."""
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    k_labels = [f"K={r['k']}" for r in reports]
    # Gather fold F1s, check both 'fold_f1s' and 'fold_f1_scores' (supporting different key naming)
    f1_data = []
    for r in reports:
        f1s = r.get("fold_f1s") or r.get("fold_f1_scores") or []
        # Convert to percentage
        f1_data.append([val * 100 for val in f1s])
        
    # Styled boxplot
    box = plt.boxplot(
        f1_data, 
        labels=k_labels, 
        patch_artist=True,
        medianprops={'color': COLORS['neutral_dark'], 'linewidth': 2},
        boxprops={'facecolor': COLORS['band_fill'], 'color': COLORS['primary'], 'linewidth': 1.5},
        whiskerprops={'color': COLORS['primary'], 'linewidth': 1.5},
        capprops={'color': COLORS['primary'], 'linewidth': 1.5},
        flierprops={'marker': 'o', 'markerfacecolor': COLORS['secondary'], 'alpha': 0.8}
    )
    
    # Add individual data points (jittered scatter) for beeswarm effect
    for i, dist in enumerate(f1_data, 1):
        y = dist
        x = np.random.normal(i, 0.04, size=len(y))
        plt.scatter(x, y, color=COLORS['primary'], alpha=0.6, edgecolors='none', s=60, label="Fold Score" if i == 1 else "")
        
    plt.title("Statistical Stability: F1-Score Distribution across K-Folds", fontsize=16, pad=15, fontweight='bold', color=COLORS['neutral_dark'])
    plt.xlabel("Multi-Fold Sweep (K-Folds)", fontsize=14, labelpad=10)
    plt.ylabel("Macro F1-Score (%)", fontsize=14, labelpad=10)
    plt.ylim(min([min(x) for x in f1_data]) - 2.0, 101.0)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[PLOT] Boxplot saved successfully to: '{output_path}'")

def plot_scaling_curve(reports: List[Dict[str, Any]], output_path: str):
    """Generates the data scaling curve showing performance rising and standard deviation shrinking."""
    plt.figure(figsize=(10, 6))
    
    k_vals = np.array([r["k"] for r in reports])
    means = np.array([r.get("mean_f1") or r.get("mean_f1_score", 0) for r in reports]) * 100
    stds = np.array([r.get("std_f1") or r.get("std_f1_score", 0) for r in reports]) * 100
    
    # Plot standard deviation band
    plt.fill_between(
        k_vals, 
        means - stds, 
        means + stds, 
        color=COLORS['band_fill'], 
        alpha=0.5, 
        label="Standard Deviation Band (±σ)"
    )
    
    # Plot mean line and scatter markers
    plt.plot(k_vals, means, color=COLORS['primary'], linewidth=2.5, marker='o', markersize=8, label="Mean F1-Score (μ)")
    
    # Annotate points with mean ± std text
    for x, y, std in zip(k_vals, means, stds):
        plt.annotate(
            f"{y:.2f}% ± {std:.2f}%",
            xy=(x, y),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS['primary'], alpha=0.8)
        )
        
    plt.title("Model Data-Scaling Curve: Swin-Tiny Convergence Profile", fontsize=16, pad=15, fontweight='bold', color=COLORS['neutral_dark'])
    plt.xlabel("K-Folds (Proportion of Training Data Scales 50% → 85.7%)", fontsize=14, labelpad=10)
    plt.ylabel("Macro F1-Score (%)", fontsize=14, labelpad=10)
    plt.xticks(k_vals)
    plt.ylim(min(means - stds) - 3.0, 101.0)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[PLOT] Data Scaling Curve saved successfully to: '{output_path}'")

def plot_multi_metrics(reports: List[Dict[str, Any]], output_path: str):
    """Generates a line comparison plot showing convergence of all four key metrics."""
    plt.figure(figsize=(10, 6))
    
    k_vals = [r["k"] for r in reports]
    
    metrics = {
        "Accuracy": ([r.get("mean_accuracy", 0) * 100 for r in reports], COLORS['primary'], 'o'),
        "Precision (Macro)": ([r.get("mean_precision", 0) * 100 for r in reports], COLORS['secondary'], 's'),
        "Recall (Macro)": ([r.get("mean_recall", 0) * 100 for r in reports], COLORS['accent'], '^'),
        "F1-Score (Macro)": ([r.get("mean_f1", 0) * 100 for r in reports], COLORS['warning'], 'D')
    }
    
    for label, (values, color, marker) in metrics.items():
        plt.plot(k_vals, values, label=label, color=color, marker=marker, markersize=8, linewidth=2.5)
        
    plt.title("Multi-Metric Cross-Validation Convergence Sweep", fontsize=16, pad=15, fontweight='bold', color=COLORS['neutral_dark'])
    plt.xlabel("K-Folds Selection", fontsize=14, labelpad=10)
    plt.ylabel("Performance Score (%)", fontsize=14, labelpad=10)
    plt.xticks(k_vals)
    plt.ylim(min([min(v[0]) for v in metrics.values()]) - 2.0, 101.0)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[PLOT] Multi-Metric Convergence Plot saved successfully to: '{output_path}'")

def plot_beeswarm_insight(reports: List[Dict[str, Any]], output_path: str):
    """Generates a scatter/beeswarm plot with standard deviation lines to expose fold stability."""
    plt.figure(figsize=(10, 6))
    
    x_data = []
    y_data = []
    
    for r in reports:
        k = r["k"]
        f1s = r.get("fold_f1s") or r.get("fold_f1_scores") or []
        for val in f1s:
            x_data.append(k)
            y_data.append(val * 100)
            
    # Draw standard deviation range lines for each K
    for r in reports:
        k = r["k"]
        mean = (r.get("mean_f1") or r.get("mean_f1_score", 0)) * 100
        std = (r.get("std_f1") or r.get("std_f1_score", 0)) * 100
        # Mean horizontal line
        plt.hlines(mean, k - 0.2, k + 0.2, colors=COLORS['neutral_dark'], linewidths=3, zorder=2)
        # Std vertical error bar
        plt.errorbar(k, mean, yerr=std, color=COLORS['secondary'], elinewidth=2.5, capsize=6, fmt='none', zorder=1)
        
    # Scatter points with jitter
    x_jitter = np.array(x_data) + np.random.uniform(-0.06, 0.06, size=len(x_data))
    plt.scatter(
        x_jitter, 
        y_data, 
        color=COLORS['primary'], 
        alpha=0.8, 
        s=80, 
        edgecolors='white', 
        linewidths=0.8, 
        zorder=3, 
        label="Individual Fold Run"
    )
    
    # Custom legend elements for clarification
    plt.scatter([], [], color=COLORS['neutral_dark'], marker='_', s=200, linewidths=3, label="Fold Mean (μ)")
    plt.errorbar([], [], yerr=1, color=COLORS['secondary'], elinewidth=2.5, capsize=6, label="Fold Deviation (±σ)")
    
    plt.title("Fold-Level Scatter: Identifying Clinical Outlier Stability", fontsize=16, pad=15, fontweight='bold', color=COLORS['neutral_dark'])
    plt.xlabel("K-Folds Selection", fontsize=14, labelpad=10)
    plt.ylabel("Macro F1-Score (%)", fontsize=14, labelpad=10)
    plt.xticks([r["k"] for r in reports])
    plt.ylim(min(y_data) - 2.0, 101.0)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[PLOT] Fold Scatter plot saved successfully to: '{output_path}'")

def main():
    args = parse_args()
    
    # Load available reports
    reports = load_kfold_data(args.reports_dir)
    
    if not reports:
        print("\n[ALERT] No K-Fold reports found to visualize.")
        print("Please copy your report JSON files into 'outputs/kfold/' first.")
        print("Example file contents required:")
        print("  swin_tiny_k2_report.json")
        print("  swin_tiny_k3_report.json")
        print("  swin_tiny_k5_report.json")
        print("  swin_tiny_k7_report.json")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("GENERATING BEAUTIFUL K-FOLD VISUALIZATION ASSETS")
    print("=" * 60)
    
    # Plot 1: Boxplot
    plot_boxplots(reports, os.path.join(args.output_dir, "kfold_boxplot.png"))
    
    # Plot 2: Scaling Curve
    plot_scaling_curve(reports, os.path.join(args.output_dir, "kfold_scaling_curve.png"))
    
    # Plot 3: Multi-Metric Convergence
    plot_multi_metrics(reports, os.path.join(args.output_dir, "kfold_multi_metrics.png"))
    
    # Plot 4: Beeswarm Fold Stability
    plot_beeswarm_insight(reports, os.path.join(args.output_dir, "kfold_fold_stability.png"))
    
    print("\n" + "=" * 60)
    print(f"SUCCESS! All K-Fold visualizations saved to: '{args.output_dir}'")
    print("=" * 60)

if __name__ == "__main__":
    main()
