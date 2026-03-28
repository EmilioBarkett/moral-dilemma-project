"""
plot_results.py
===============
Analysis and visualisation for experiment-v3 results.

Reads all *_summary.csv files from a folder (default: results/) and produces:
  1. overall_chart.png  — congruent vs incongruent harm endorsement per model
  2. foundation_heatmap.png — endorsement gap per model × moral foundation

Also prints a summary table to the console.

USAGE
-----
  python plot_results.py
  python plot_results.py --folder results --output-dir plots
"""

import argparse
import csv
import glob
import math
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Model display labels (OpenRouter ID → short name for charts)
# ---------------------------------------------------------------------------
MODEL_LABELS = {
    # OpenAI
    "openai/gpt-4o":                        "GPT-4o",
    "openai/gpt-4o-mini":                   "GPT-4o-mini",
    "openai/gpt-5":                         "GPT-5",
    "openai/o1":                            "o1",
    "openai/o3":                            "o3",
    # Anthropic
    "anthropic/claude-3.5-sonnet":          "Claude 3.5 Sonnet",
    "anthropic/claude-opus-4":              "Claude Opus 4",
    # Google
    "google/gemini-2.5-pro":               "Gemini 2.5 Pro",
    "google/gemma-3-27b-it":               "Gemma 3 27B",
    # xAI
    "x-ai/grok-3":                         "Grok 3",
    "x-ai/grok-4":                         "Grok 4",
    # Amazon
    "amazon/nova-premier-v1":              "Nova Premier",
    # Microsoft
    "microsoft/phi-4":                     "Phi-4",
    # Meta
    "meta-llama/llama-3.3-70b-instruct":   "Llama 3.3 70B",
    "meta-llama/llama-4-maverick":         "Llama 4 Maverick",
    # Mistral
    "mistralai/mixtral-8x22b-instruct":    "Mixtral 8x22B",
    "mistralai/mistral-large-2512":        "Mistral Large 3",
    # DeepSeek
    "deepseek/deepseek-r1":               "DeepSeek R1",
    "deepseek/deepseek-r1-0528":          "DeepSeek R1-0528",
    "deepseek/deepseek-chat-v3-0324":     "DeepSeek V3",
    # Qwen
    "qwen/qwen3-235b-a22b":               "Qwen3 235B",
    "qwen/qwen3-235b-a22b-thinking-2507": "Qwen3 235B (Think)",
    # Chinese labs
    "z-ai/glm-4.7":                       "GLM-4.7",
    "baidu/ernie-4.5-300b-a47b":          "Ernie 4.5",
    # Other
    "upstage/solar-pro-3":                "Solar Pro 3",
    "cohere/command-a":                   "Command A",
}

# Which models use extended chain-of-thought reasoning
REASONING_MODELS = {
    "openai/o1", "openai/o3",
    "anthropic/claude-opus-4",
    "google/gemini-2.5-pro",
    "x-ai/grok-3",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-0528",
    "qwen/qwen3-235b-a22b-thinking-2507",
}

# Human baseline from Conway & Gawronski (2013)
HUMAN_BASELINE = {
    "label":        "Human\n(C&G 2013)",
    "cong_mean":    25.38,
    "cong_se":      1.50,
    "incong_mean":  56.08,
    "incong_se":    1.50,
    "is_human":     True,
}

FOUNDATIONS = ["authority", "care", "fairness", "loyalty", "purity"]
FOUNDATION_LABELS = {
    "authority": "Authority",
    "care":      "Care",
    "fairness":  "Fairness",
    "loyalty":   "Loyalty",
    "purity":    "Purity",
}

COLOR_CONGRUENT   = "#E8756A"   # coral
COLOR_INCONGRUENT = "#63C6C4"   # teal
COLOR_HUMAN_C     = "#C94040"   # darker coral for human
COLOR_HUMAN_I     = "#3A9A98"   # darker teal for human


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _to_float(v):
    try:
        f = float(v)
        return float("nan") if math.isnan(f) else f
    except (ValueError, TypeError):
        return float("nan")


def load_summaries(folder):
    pattern = os.path.join(folder, "*_summary.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No *_summary.csv files found in '{folder}'. "
            "Run the experiment first."
        )
    overall    = {}   # model_id -> row dict
    foundation = {}   # model_id -> {foundation -> row dict}

    for fpath in files:
        with open(fpath, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                model = row["model"]
                level = row.get("level", "overall")
                f     = row.get("foundation", "all")

                entry = {
                    "model":        model,
                    "label":        MODEL_LABELS.get(model, model),
                    "reasoning":    model in REASONING_MODELS,
                    "n_runs":       _to_float(row.get("n_runs", "")),
                    "cong_mean":    _to_float(row["mean_harm_endorsement_congruent"]),
                    "cong_se":      _to_float(row["se_harm_endorsement_congruent"]),
                    "incong_mean":  _to_float(row["mean_harm_endorsement_incongruent"]),
                    "incong_se":    _to_float(row["se_harm_endorsement_incongruent"]),
                }

                if level == "overall" or f == "all":
                    if model not in overall:
                        overall[model] = entry
                else:
                    if model not in foundation:
                        foundation[model] = {}
                    foundation[model][f] = entry

    return overall, foundation


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def print_table(overall):
    entries = sorted(overall.values(), key=lambda e: (
        e["incong_mean"] - e["cong_mean"]
    ), reverse=True)

    header = f"{'Model':<28} {'R':>2}  {'Cong%':>6}  {'Incong%':>8}  {'Gap':>6}  {'N':>4}"
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")
    for e in entries:
        gap = e["incong_mean"] - e["cong_mean"]
        r   = "Y" if e["reasoning"] else " "
        n   = int(e["n_runs"]) if not math.isnan(e["n_runs"]) else "-"
        print(
            f"{e['label']:<28} {r:>2}  "
            f"{e['cong_mean']:>6.1f}  {e['incong_mean']:>8.1f}  "
            f"{gap:>6.1f}  {n:>4}"
        )

    # Human baseline
    print(f"{'─'*len(header)}")
    gap_h = HUMAN_BASELINE["incong_mean"] - HUMAN_BASELINE["cong_mean"]
    print(
        f"{'Human (C&G 2013)':<28} {' ':>2}  "
        f"{HUMAN_BASELINE['cong_mean']:>6.1f}  "
        f"{HUMAN_BASELINE['incong_mean']:>8.1f}  "
        f"{gap_h:>6.1f}  {'—':>4}"
    )
    print(f"{'='*len(header)}\n")
    print("R = Reasoning model (extended chain-of-thought)\n")


# ---------------------------------------------------------------------------
# Plot 1: Overall bar chart
# ---------------------------------------------------------------------------

def plot_overall(overall, output_path):
    # Sort models by gap descending; human baseline prepended separately
    entries = sorted(
        overall.values(),
        key=lambda e: (e["incong_mean"] - e["cong_mean"]),
        reverse=True,
    )
    all_entries = [HUMAN_BASELINE] + entries

    labels       = [e["label"]       for e in all_entries]
    cong_means   = [e["cong_mean"]   for e in all_entries]
    cong_ses     = [e["cong_se"]     for e in all_entries]
    incong_means = [e["incong_mean"] for e in all_entries]
    incong_ses   = [e["incong_se"]   for e in all_entries]
    is_human     = [e.get("is_human", False) for e in all_entries]
    is_reasoning = [e.get("reasoning", False) for e in all_entries]

    n      = len(all_entries)
    x      = np.arange(n)
    width  = 0.35
    offset = width / 2

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(max(14, n * 1.1), 7))

    c_colors = [COLOR_HUMAN_C if h else COLOR_CONGRUENT   for h in is_human]
    i_colors = [COLOR_HUMAN_I if h else COLOR_INCONGRUENT for h in is_human]

    bars_c = ax.bar(
        x - offset, cong_means, width=width, color=c_colors,
        yerr=cong_ses, capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": "black"},
        zorder=3,
    )
    bars_i = ax.bar(
        x + offset, incong_means, width=width, color=i_colors,
        yerr=incong_ses, capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": "black"},
        zorder=3,
    )

    # Mark reasoning models with an asterisk on x-axis label
    tick_labels = []
    for label, reasoning in zip(labels, is_reasoning):
        tick_labels.append(f"{label} *" if reasoning else label)

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("% Harm Endorsement", fontsize=12)
    ax.set_title(
        "Harm Endorsement: Congruent vs Incongruent Dilemmas\n"
        "(sorted by moral sensitivity gap; * = reasoning model)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, 115)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    legend_patches = [
        mpatches.Patch(color=COLOR_CONGRUENT,   label="Congruent (low stakes)"),
        mpatches.Patch(color=COLOR_INCONGRUENT, label="Incongruent (high stakes)"),
        mpatches.Patch(color=COLOR_HUMAN_C,     label="Human baseline (C&G 2013)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", frameon=True, fontsize=10)

    sns.despine()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved: '{output_path}'")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2: Foundation gap heatmap
# ---------------------------------------------------------------------------

def plot_foundation_heatmap(overall, foundation, output_path):
    # Same model order as overall chart (sorted by gap)
    model_order = sorted(
        overall.keys(),
        key=lambda m: (overall[m]["incong_mean"] - overall[m]["cong_mean"]),
        reverse=True,
    )
    model_labels = [overall[m]["label"] for m in model_order]
    reasoning    = [overall[m]["reasoning"] for m in model_order]

    # Build gap matrix (models × foundations)
    matrix = []
    for m in model_order:
        row = []
        for f in FOUNDATIONS:
            fd = foundation.get(m, {}).get(f)
            if fd:
                gap = fd["incong_mean"] - fd["cong_mean"]
                row.append(gap if not math.isnan(gap) else float("nan"))
            else:
                row.append(float("nan"))
        matrix.append(row)

    matrix_np = np.array(matrix, dtype=float)

    # X-axis tick labels with asterisk for reasoning models
    y_labels = [
        f"{lbl} *" if r else lbl
        for lbl, r in zip(model_labels, reasoning)
    ]
    x_labels = [FOUNDATION_LABELS[f] for f in FOUNDATIONS]

    sns.set_theme(style="white", font_scale=1.05)
    fig_h = max(6, len(model_order) * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(7, fig_h))

    mask = np.isnan(matrix_np)
    sns.heatmap(
        matrix_np,
        ax=ax,
        mask=mask,
        cmap="RdYlGn",
        center=0,
        vmin=-30, vmax=100,
        annot=True, fmt=".0f", annot_kws={"size": 8},
        linewidths=0.4, linecolor="white",
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar_kws={"label": "Incongruent − Congruent gap (pp)", "shrink": 0.7},
    )
    ax.set_title(
        "Moral Sensitivity Gap by Foundation\n"
        "(Incongruent − Congruent; higher = more sensitive; * = reasoning)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Moral Foundation", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: '{output_path}'")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--folder",     "-f", default="results",
                   help="Folder containing *_summary.csv files.")
    p.add_argument("--output-dir", "-o", default="plots",
                   help="Directory to save output PNGs.")
    p.add_argument("--no-heatmap",       action="store_true",
                   help="Skip the foundation heatmap (useful if only one model is loaded).")
    args = p.parse_args()

    overall, foundation = load_summaries(args.folder)
    print(f"Loaded {len(overall)} model(s): {[v['label'] for v in overall.values()]}")

    print_table(overall)

    plot_overall(
        overall,
        os.path.join(args.output_dir, "overall_chart.png"),
    )

    if not args.no_heatmap and foundation:
        plot_foundation_heatmap(
            overall, foundation,
            os.path.join(args.output_dir, "foundation_heatmap.png"),
        )
    elif not args.no_heatmap:
        print("Note: no foundation-level data found — skipping heatmap. "
              "Run without --simple to collect foundation data.")


if __name__ == "__main__":
    main()
