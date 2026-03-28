"""
plot_results.py
===============
Generates a bar chart of % Harm Endorsement (congruent vs incongruent)
across models, matching the style of Conway & Gawronski (2013) replication
figures.

Reads all *_summary.csv files from a folder (default: "summaries"),
plus a fixed human baseline, and saves a PNG.

USAGE
-----
  python plot_results.py
  python plot_results.py --folder summaries --output harm_endorsement.png
"""

import argparse
import glob
import os
import math
import csv

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Edit this dictionary to control x-axis labels.
# Key = model string in the summary CSV, Value = display label.
# Any model not listed here will use the raw string from the CSV.
# ---------------------------------------------------------------------------
MODEL_LABELS = {
    "openai/gpt-4o":                    "GPT-4o",
    "deepseek/deepseek-chat-v3-0324":   "DeepSeek V3",
    "x-ai/grok-3":                      "Grok-3",
    "x-ai/grok-4":                      "Grok-4",
}

# Human baseline from Conway & Gawronski (2013)
HUMAN_BASELINE = {
    "model":                                "Human",
    "mean_harm_endorsement_congruent":      25.38,
    "se_harm_endorsement_congruent":        1.50,   # approximate from figure
    "mean_harm_endorsement_incongruent":    56.08,
    "se_harm_endorsement_incongruent":      1.50,
}

# Colors matching the reference figure
COLOR_CONGRUENT   = "#E8756A"   # coral/salmon
COLOR_INCONGRUENT = "#63C6C4"   # teal


def load_summaries(folder):
    pattern = os.path.join(folder, "*_summary.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No *_summary.csv files found in '{folder}'. "
            "Check the folder path or run the experiment first."
        )
    rows = []
    for fpath in files:
        with open(fpath, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append({k: v for k, v in row.items()})
    return rows


def clean_row(row):
    """Convert numeric strings to floats, handle nan."""
    def to_float(v):
        try:
            f = float(v)
            return float("nan") if math.isnan(f) else f
        except (ValueError, TypeError):
            return float("nan")
    return {
        "model":        row["model"],
        "label":        MODEL_LABELS.get(row["model"], row["model"]),
        "cong_mean":    to_float(row["mean_harm_endorsement_congruent"]),
        "cong_se":      to_float(row["se_harm_endorsement_congruent"]),
        "incong_mean":  to_float(row["mean_harm_endorsement_incongruent"]),
        "incong_se":    to_float(row["se_harm_endorsement_incongruent"]),
    }


def plot(data, output_path):
    # Human baseline first, then models in file order
    human = {
        "label":       "Human",
        "cong_mean":   HUMAN_BASELINE["mean_harm_endorsement_congruent"],
        "cong_se":     HUMAN_BASELINE["se_harm_endorsement_congruent"],
        "incong_mean": HUMAN_BASELINE["mean_harm_endorsement_incongruent"],
        "incong_se":   HUMAN_BASELINE["se_harm_endorsement_incongruent"],
    }
    entries = [human] + data

    labels      = [e["label"]       for e in entries]
    cong_means  = [e["cong_mean"]   for e in entries]
    cong_ses    = [e["cong_se"]     for e in entries]
    incong_means= [e["incong_mean"] for e in entries]
    incong_ses  = [e["incong_se"]   for e in entries]

    n      = len(entries)
    x      = list(range(n))
    width  = 0.35
    offset = width / 2

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(2.2 * n + 1.5, 7))

    bars_c = ax.bar(
        [xi - offset for xi in x], cong_means,
        width=width, color=COLOR_CONGRUENT,
        yerr=cong_ses, capsize=4,
        label="Harm Endorsement (Congruent)",
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )
    bars_i = ax.bar(
        [xi + offset for xi in x], incong_means,
        width=width, color=COLOR_INCONGRUENT,
        yerr=incong_ses, capsize=4,
        label="Harm Endorsement (Incongruent)",
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )

    # Bar value labels
    def label_bars(bars, means):
        for bar, mean in zip(bars, means):
            if not math.isnan(mean):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{mean:.2f}",
                    ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.6),
                )

    label_bars(bars_c, cong_means)
    label_bars(bars_i, incong_means)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("% Harm Endorsement", fontsize=12)
    ax.set_title("Moral Dilemmas: Harm Endorsement Scores", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(incong_means + cong_means) * 1.25 + 10)
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlabel("Model / Group", fontsize=12)

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: '{output_path}'")
    plt.close()


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--folder",  "-f", default="summaries", help="Folder containing *_summary.csv files.")
    p.add_argument("--output",  "-o", default="harm_endorsement.png", help="Output PNG filename.")
    args = p.parse_args()

    rows = load_summaries(args.folder)
    data = [clean_row(r) for r in rows]
    print(f"Loaded {len(data)} model(s): {[d['label'] for d in data]}")
    plot(data, args.output)


if __name__ == "__main__":
    main()
