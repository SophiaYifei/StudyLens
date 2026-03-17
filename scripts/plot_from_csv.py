"""
plot_from_csv.py - Generate all evaluation plots for the StudyLens report.

Reads data/eval/evaluation_averages.csv,
produces publication-quality figures saved to data/eval/plots/.

Usage:
    python scripts/plot_from_csv.py

AI Attribution: Code co-authored with Claude (Anthropic, https://claude.ai)
for structural design, debugging, and documentation.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


COLORS = {
    "naive_first5":  "#9e9e9e",
    "naive_random":  "#bdbdbd",
    "tfidf":         "#607d8b",
    "bart":          "#42a5f5",
    "bart-samsum":   "#66bb6a",
    "led-arxiv":     "#ef5350",
    "longt5":        "#ab47bc",
    "qwen7b":        "#ff9800",
    "qwen7b-ft":     "#e91e63",
}


def load_data():
    root = Path("data") / "eval"
    avg = pd.read_csv(root / "evaluation_averages.csv")
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return avg, plots_dir


def _make_label(row):
    m = str(row["model"])
    s = str(row.get("strategy", ""))
    r = str(row.get("ratio", ""))
    if r and r != "nan" and r != "":
        return f"{m} ({r})"
    if m in ("tfidf", "naive_first5", "naive_random", "qwen7b-ft"):
        return m
    if s in ("concat", "final"):
        return f"{m} ({s})"
    return m


# ═══════════════════════════════════════════════════════════════
# Fig 1-3: Horizontal bar charts for each metric
# ═══════════════════════════════════════════════════════════════

def plot_metric_bars(avg_df, metric, ylabel, title, filename, plots_dir):
    df = avg_df.copy()
    df["label"] = df.apply(_make_label, axis=1)
    df = df.sort_values(metric, ascending=True)

    n = len(df)
    fig_height = max(6, n * 0.38)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    colors = [COLORS.get(str(row["model"]), "#888") for _, row in df.iterrows()]
    y_pos = np.arange(n)

    bars = ax.barh(y_pos, df[metric].values, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"].values, fontsize=9)
    ax.set_xlabel(ylabel)
    ax.set_title(title, pad=12)
    ax.axvline(x=0, color="black", linewidth=0.5)

    for bar, val in zip(bars, df[metric].values):
        x_pos = val + 0.004 if val >= 0 else val - 0.004
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha=ha, fontsize=8, fontweight="bold")

    plt.tight_layout()
    out = plots_dir / filename
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════
# Fig 4: Qwen ratio hyperparameter tuning
# ═══════════════════════════════════════════════════════════════

def plot_qwen_ratio_tuning(avg_df, plots_dir):
    qwen = avg_df[avg_df["model"] == "qwen7b"].copy()
    if qwen.empty:
        return

    qwen["ratio_num"] = qwen["ratio"].str.replace("ratio", "").astype(int) / 100
    qwen = qwen.sort_values("ratio_num")

    fig, ax = plt.subplots(figsize=(8, 5))

    lines = [
        ("rouge_l_f1",          "ROUGE-L F1",     "#2196f3", "o"),
        ("bertscore_f1",        "BERTScore F1",   "#4caf50", "s"),
        ("avg_entailment_score","NLI Entailment",  "#ff9800", "^"),
    ]

    for col, label, color, marker in lines:
        if col in qwen.columns:
            ax.plot(qwen["ratio_num"], qwen[col], marker=marker, color=color,
                    label=label, linewidth=2.5, markersize=8)

    ax.set_xlabel("Output Length Ratio")
    ax.set_ylabel("Score")
    ax.set_title("Qwen2.5-7B: Effect of Output Length Ratio on Metrics")
    ax.legend()
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xticks(qwen["ratio_num"].values)

    plt.tight_layout()
    out = plots_dir / "qwen_ratio_tuning.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════
# Fig 5: Ablation concat vs final
# ═══════════════════════════════════════════════════════════════

def plot_ablation(avg_df, plots_dir):
    models = ["bart", "longt5", "bart-samsum", "led-arxiv"]
    metrics_info = [
        ("rouge_l_f1",           "ROUGE-L F1"),
        ("bertscore_f1",         "BERTScore F1"),
        ("avg_entailment_score", "NLI Entailment"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(models))
    width = 0.35

    for ax, (metric, ylabel) in zip(axes, metrics_info):
        if metric not in avg_df.columns:
            continue
        c_vals, f_vals = [], []
        for m in models:
            cr = avg_df[(avg_df["model"] == m) & (avg_df["strategy"] == "concat")]
            fr = avg_df[(avg_df["model"] == m) & (avg_df["strategy"] == "final")]
            c_vals.append(cr[metric].iloc[0] if not cr.empty else 0)
            f_vals.append(fr[metric].iloc[0] if not fr.empty else 0)

        ax.bar(x - width/2, c_vals, width, label="concat", color="#42a5f5")
        ax.bar(x + width/2, f_vals, width, label="final",  color="#ef5350")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=9)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    plt.suptitle("Ablation: Concat vs. Final-Pass Summarization", fontsize=14, y=1.02)
    plt.tight_layout()
    out = plots_dir / "ablation_concat_final.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════
# Fig 6: Radar chart
# ═══════════════════════════════════════════════════════════════

def plot_radar(avg_df, plots_dir):
    selected = [
        ("naive_first5", "first5",  None),
        ("tfidf",        "tfidf",   None),
        ("bart",         "concat",  None),
        ("longt5",       "concat",  None),
        ("qwen7b",       "ratio",   "ratio06"),
        ("qwen7b-ft",    "final",   None),
    ]

    metrics = ["rouge_l_f1", "avg_entailment_score", "bertscore_f1"]
    metric_labels = ["ROUGE-L F1", "NLI Entailment", "BERTScore F1"]

    # Distinct, high-contrast colors
    radar_colors = [
        "#9e9e9e",   # naive_first5 - gray
        "#1565c0",   # tfidf - dark blue
        "#43a047",   # bart - green
        "#8e24aa",   # longt5 - purple
        "#ff9800",   # qwen7b - orange
        "#e91e63",   # qwen7b-ft - pink/magenta
    ]

    labels = []
    values_list = []
    colors_used = []

    for i, (model, strategy, ratio) in enumerate(selected):
        mask = (avg_df["model"] == model) & (avg_df["strategy"] == strategy)
        if ratio:
            mask = mask & (avg_df["ratio"] == ratio)
        row = avg_df[mask]
        if row.empty:
            continue
        label = model if ratio is None else f"{model} ({ratio})"
        labels.append(label)
        values_list.append([row[m].iloc[0] for m in metrics])
        colors_used.append(radar_colors[i])

    if not values_list:
        return

    # Normalize to [0,1]
    arr = np.array(values_list)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normed = (arr - mins) / ranges

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, (label, nvals) in enumerate(zip(labels, normed)):
        vals = nvals.tolist() + [nvals[0]]
        c = colors_used[i]
        ax.plot(angles, vals, linewidth=2.5, label=label, color=c)
        ax.fill(angles, vals, alpha=0.06, color=c)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Multi-Metric Comparison (normalized)", pad=25, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.12), fontsize=10)

    plt.tight_layout()
    out = plots_dir / "radar_top_models.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_finetune_train_test(avg_csv_path, plots_dir):
    """Fig 7: Compare qwen7b-ft on train vs test topics."""
    per_file = pd.read_csv(avg_csv_path.parent / "evaluation_results_all.csv")
    
    ft = per_file[per_file["model"] == "qwen7b-ft"].copy()
    if ft.empty:
        print("  No qwen7b-ft data found.")
        return

    test_topics = {"dl_s5", "ml_s5"}
    ft["split"] = ft["topic_key"].apply(lambda x: "Test (unseen)" if x in test_topics else "Train")

    metrics = ["rouge_l_f1", "bertscore_f1"]
    labels = ["ROUGE-L F1", "BERTScore F1"]
    if "avg_entailment_score" in ft.columns:
        metrics.append("avg_entailment_score")
        labels.append("NLI Entailment")

    grouped = ft.groupby("split")[metrics].mean()

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    train_vals = grouped.loc["Train"].values
    test_vals = grouped.loc["Test (unseen)"].values

    ax.bar(x - width/2, train_vals, width, label="Train (8 topics)", color="#42a5f5")
    ax.bar(x + width/2, test_vals, width, label="Test (2 unseen topics)", color="#e91e63")

    for i, (tv, uv) in enumerate(zip(train_vals, test_vals)):
        ax.text(i - width/2, tv + 0.005, f"{tv:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + width/2, uv + 0.005, f"{uv:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Qwen2.5-7B Fine-Tuned: Train vs. Test (Unseen) Topics")
    ax.legend()
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    out = plots_dir / "finetune_train_vs_test.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

# ═══════════════════════════════════════════════════════════════

def main():
    plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    })

    avg_df, plots_dir = load_data()
    print("Generating plots...\n")

    # Fig 1: ROUGE-L
    plot_metric_bars(avg_df, "rouge_l_f1", "ROUGE-L F1",
                     "ROUGE-L F1 by Model Configuration (averaged across topics)",
                     "all_models_rouge.png", plots_dir)

    # Fig 2: BERTScore
    plot_metric_bars(avg_df, "bertscore_f1", "BERTScore F1",
                     "BERTScore F1 by Model Configuration (averaged across topics)",
                     "all_models_bertscore.png", plots_dir)

    # Fig 3: NLI
    if "avg_entailment_score" in avg_df.columns:
        plot_metric_bars(avg_df, "avg_entailment_score", "NLI Entailment Score",
                         "NLI Entailment by Model Configuration (averaged across topics)",
                         "all_models_nli.png", plots_dir)

    # Fig 4: Qwen ratio tuning
    plot_qwen_ratio_tuning(avg_df, plots_dir)

    # Fig 5: Ablation
    plot_ablation(avg_df, plots_dir)

    # Fig 6: Radar
    plot_radar(avg_df, plots_dir)

    # Fig 7: Fine-tune train vs test
    plot_finetune_train_test(plots_dir.parent / "evaluation_averages.csv", plots_dir)

    print(f"\nAll plots saved to {plots_dir}")


if __name__ == "__main__":
    main()