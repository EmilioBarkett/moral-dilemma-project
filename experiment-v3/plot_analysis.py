"""
plot_analysis.py
================
Generates three analysis plots for the Study 1 moral dilemma results:

  1. overall_chart.png        — all models sorted by gap, reasoning highlighted
  2. within_lab_contrasts.png — paired bars per lab (reasoning vs non-reasoning)
  3. foundation_gaps.png      — mean gap per moral foundation (pooled across models)

USAGE
-----
  python plot_analysis.py --folder results --output-dir plots
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
# Metadata
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    "openai/gpt-3.5-turbo":                  "GPT-3.5 Turbo",
    "openai/gpt-4o":                         "GPT-4o",
    "openai/gpt-4o-mini":                    "GPT-4o-mini",
    "openai/gpt-5-chat":                     "GPT-5",
    "openai/gpt-5":                          "GPT-5",
    "openai/o1":                             "o1",
    "openai/o3":                             "o3",
    "anthropic/claude-3.5-sonnet":           "Claude 3.5 Sonnet",
    "anthropic/claude-opus-4":               "Claude Opus 4",
    "google/gemini-2.5-pro":                 "Gemini 2.5 Pro",
    "google/gemini-2.0-flash-001":           "Gemini 2.0 Flash",
    "google/gemma-3-27b-it":                 "Gemma 3 27B",
    "x-ai/grok-3":                           "Grok 3",
    "x-ai/grok-4":                           "Grok 4",
    "amazon/nova-premier-v1":                "Nova Premier",
    "microsoft/phi-4":                       "Phi-4",
    "meta-llama/llama-3.2-3b-instruct":      "Llama 3.2 3B",
    "meta-llama/llama-3.3-70b-instruct":     "Llama 3.3 70B",
    "meta-llama/llama-4-maverick":           "Llama 4 Maverick",
    "mistralai/mixtral-8x22b-instruct":      "Mixtral 8x22B",
    "mistralai/mistral-large-2512":          "Mistral Large 3",
    "deepseek/deepseek-chat-v3-0324":        "DeepSeek V3",
    "deepseek/deepseek-r1":                  "DeepSeek R1",
    "deepseek/deepseek-r1-0528":             "DeepSeek R1-0528",
    "qwen/qwen3-235b-a22b":                  "Qwen3 235B",
    "qwen/qwen3-235b-a22b-thinking-2507":    "Qwen3 235B\n(Thinking)",
    "baidu/ernie-4.5-300b-a47b":             "Ernie 4.5",
    "amazon/nova-premier-v1":                "Nova Premier",
    "cohere/command-a":                      "Command A",
    "upstage/solar-pro-3":                   "Solar Pro 3",
}

REASONING_MODELS = {
    "openai/o1", "openai/o3",
    "anthropic/claude-opus-4",
    "google/gemini-2.5-pro",
    "x-ai/grok-3",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-0528",
    "qwen/qwen3-235b-a22b-thinking-2507",
}

WITHIN_LAB_PAIRS = [
    ("Anthropic",  "anthropic/claude-3.5-sonnet",     "anthropic/claude-opus-4"),
    ("xAI",        "x-ai/grok-4",                     "x-ai/grok-3"),
    ("OpenAI",     "openai/gpt-4o",                   "openai/o1"),
    ("OpenAI",     "openai/gpt-5-chat",               "openai/o3"),
    ("DeepSeek",   "deepseek/deepseek-chat-v3-0324",  "deepseek/deepseek-r1"),
    ("DeepSeek",   "deepseek/deepseek-chat-v3-0324",  "deepseek/deepseek-r1-0528"),
    ("Qwen",       "qwen/qwen3-235b-a22b",            "qwen/qwen3-235b-a22b-thinking-2507"),
    ("Google",     "google/gemma-3-27b-it",           "google/gemini-2.5-pro"),
]

FOUNDATIONS    = ["authority", "care", "fairness", "loyalty", "purity"]
FOUND_LABELS   = {"authority": "Authority", "care": "Care",
                  "fairness": "Fairness", "loyalty": "Loyalty", "purity": "Purity"}
EXCLUDE_MODELS = {"meta-llama/llama-3.2-3b-instruct"}   # capacity failure

HUMAN_CONG   = 25.38
HUMAN_INCONG = 56.08
HUMAN_GAP    = HUMAN_INCONG - HUMAN_CONG   # 30.70

COLOR_NR   = "#5B9BD5"   # blue — non-reasoning
COLOR_R    = "#F4A261"   # orange — reasoning
COLOR_CONG = "#E8756A"   # coral — congruent bars
COLOR_INC  = "#63C6C4"   # teal — incongruent bars
COLOR_HUM  = "#555555"   # grey — human reference

sns.set_theme(style="whitegrid", font_scale=1.05)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _f(v):
    try:
        x = float(v)
        return float("nan") if math.isnan(x) else x
    except (ValueError, TypeError):
        return float("nan")


def load_summaries(folder):
    pattern = os.path.join(folder, "**", "*_summary.csv")
    files   = sorted(glob.glob(pattern, recursive=True))
    overall    = {}
    foundation = {}
    for fpath in files:
        with open(fpath, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                mid   = row["model"]
                level = row.get("level", "overall")
                fnd   = row.get("foundation", "all")
                entry = {
                    "model":       mid,
                    "label":       MODEL_LABELS.get(mid, mid),
                    "reasoning":   mid in REASONING_MODELS,
                    "n_runs":      _f(row.get("n_runs", "")),
                    "cong_mean":   _f(row["mean_harm_endorsement_congruent"]),
                    "cong_se":     _f(row["se_harm_endorsement_congruent"]),
                    "incong_mean": _f(row["mean_harm_endorsement_incongruent"]),
                    "incong_se":   _f(row["se_harm_endorsement_incongruent"]),
                }
                if level == "overall" or fnd == "all":
                    if mid not in overall:
                        overall[mid] = entry
                else:
                    foundation.setdefault(mid, {})[fnd] = entry
    return overall, foundation


# ---------------------------------------------------------------------------
# Plot 1: Updated overall bar chart
# ---------------------------------------------------------------------------

def plot_overall(overall, output_path):
    entries = sorted(
        [e for mid, e in overall.items()],
        key=lambda e: e["incong_mean"] - e["cong_mean"],
        reverse=True,
    )

    n     = len(entries)
    x     = np.arange(n)
    w     = 0.35

    fig, ax = plt.subplots(figsize=(max(16, n * 1.1), 7))

    c_colors = [COLOR_CONG] * n
    i_colors = [COLOR_INC]  * n

    ax.bar(x - w/2, [e["cong_mean"]   for e in entries], width=w,
           color=c_colors, yerr=[e["cong_se"]   for e in entries],
           capsize=3, error_kw={"elinewidth": 1, "ecolor": "black"}, zorder=3)
    ax.bar(x + w/2, [e["incong_mean"] for e in entries], width=w,
           color=i_colors, yerr=[e["incong_se"] for e in entries],
           capsize=3, error_kw={"elinewidth": 1, "ecolor": "black"}, zorder=3)

    # Human baseline dotted lines
    ax.axhline(HUMAN_CONG,   color=COLOR_CONG,  linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"Human congruent ({HUMAN_CONG:.1f}%)")
    ax.axhline(HUMAN_INCONG, color=COLOR_INC,   linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"Human incongruent ({HUMAN_INCONG:.1f}%)")

    # Shade reasoning models
    for i, e in enumerate(entries):
        if e["reasoning"]:
            ax.axvspan(i - 0.5, i + 0.5, color=COLOR_R, alpha=0.08, zorder=0)

    labels = [f"{e['label']} *" if e["reasoning"] else e["label"] for e in entries]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel("% Harm Endorsement", fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title(
        "Harm Endorsement by Model: Congruent vs. Incongruent Dilemmas\n"
        "(sorted by moral sensitivity gap; * = reasoning model; shaded = reasoning)",
        fontsize=13, fontweight="bold",
    )

    legend_handles = [
        mpatches.Patch(color=COLOR_CONG, label="Congruent"),
        mpatches.Patch(color=COLOR_INC,  label="Incongruent"),
        plt.Line2D([0], [0], color=COLOR_CONG, linestyle="--", label="Human congruent baseline"),
        plt.Line2D([0], [0], color=COLOR_INC,  linestyle="--", label="Human incongruent baseline"),
        mpatches.Patch(color=COLOR_R, alpha=0.2, label="Reasoning model (shaded)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, frameon=True)

    sns.despine()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: '{output_path}'")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2: Within-lab contrasts
# ---------------------------------------------------------------------------

def plot_within_lab(overall, output_path):
    pairs = [(lab, nr, r) for lab, nr, r in WITHIN_LAB_PAIRS
             if nr in overall and r in overall]

    if not pairs:
        print("No within-lab pairs found — skipping contrast plot.")
        return

    n_pairs = len(pairs)
    fig, ax = plt.subplots(figsize=(max(10, n_pairs * 1.4), 6))

    x     = np.arange(n_pairs)
    w     = 0.35
    labs  = []
    nr_gaps, r_gaps = [], []

    for lab, nr_id, r_id in pairs:
        nr = overall[nr_id]
        r  = overall[r_id]
        nr_gap = nr["incong_mean"] - nr["cong_mean"]
        r_gap  = r["incong_mean"]  - r["cong_mean"]
        nr_gaps.append(nr_gap)
        r_gaps.append(r_gap)
        labs.append(f"{MODEL_LABELS.get(nr_id, nr_id)}\nvs.\n{MODEL_LABELS.get(r_id, r_id)}")

    bars_nr = ax.bar(x - w/2, nr_gaps, width=w, color=COLOR_NR, label="Non-reasoning", zorder=3)
    bars_r  = ax.bar(x + w/2, r_gaps,  width=w, color=COLOR_R,  label="Reasoning",     zorder=3)

    # Annotate delta
    for i, (nr_g, r_g) in enumerate(zip(nr_gaps, r_gaps)):
        delta = r_g - nr_g
        sign  = "+" if delta >= 0 else ""
        color = "#c0392b" if delta > 0 else "#27ae60"
        ax.text(i, max(nr_g, r_g) + 1.5, f"{sign}{delta:.1f}pp",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color)

    # Human gap reference
    ax.axhline(HUMAN_GAP, color=COLOR_HUM, linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"Human gap ({HUMAN_GAP:.1f} pp)")

    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("Moral Sensitivity Gap (pp)", fontsize=12)
    ax.set_ylim(0, 85)
    ax.set_title(
        "Within-Lab Reasoning vs. Non-Reasoning Contrasts\n"
        "(Δ = reasoning gap − non-reasoning gap; red = increased, green = reduced)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, frameon=True)
    sns.despine()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: '{output_path}'")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Foundation gaps (pooled across models)
# ---------------------------------------------------------------------------

def plot_foundation_gaps(overall, foundation, output_path):
    from collections import defaultdict

    f_gaps = defaultdict(list)
    for mid, fnd_data in foundation.items():
        if mid in EXCLUDE_MODELS:
            continue
        for f in FOUNDATIONS:
            fd = fnd_data.get(f)
            if fd is None:
                continue
            gap = fd["incong_mean"] - fd["cong_mean"]
            if not math.isnan(gap):
                f_gaps[f].append(gap)

    means = {f: np.mean(v) for f, v in f_gaps.items()}
    ses   = {f: np.std(v, ddof=1) / math.sqrt(len(v)) for f, v in f_gaps.items()}

    # Sort foundations by mean gap descending
    sorted_f = sorted(FOUNDATIONS, key=lambda f: means[f], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    x      = np.arange(len(sorted_f))
    colors = ["#E07B54", "#5B9BD5", "#F4A261", "#6BBF8E", "#A78BCC"]

    bars = ax.bar(x, [means[f] for f in sorted_f],
                  yerr=[ses[f] for f in sorted_f],
                  color=colors, capsize=5,
                  error_kw={"elinewidth": 1.5, "ecolor": "black"},
                  zorder=3)

    # Annotate each bar with mean ± SE
    for bar, f in zip(bars, sorted_f):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ses[f] + 1.2,
                f"{means[f]:.1f}pp", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Human gap reference
    ax.axhline(HUMAN_GAP, color=COLOR_HUM, linestyle="--", linewidth=1.5,
               alpha=0.8, label=f"Human gap baseline ({HUMAN_GAP:.1f} pp)")

    ax.set_xticks(x)
    ax.set_xticklabels([FOUND_LABELS[f] for f in sorted_f], fontsize=12)
    ax.set_ylabel("Mean Moral Sensitivity Gap (pp)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title(
        "Mean Moral Sensitivity Gap by Foundation\n"
        "(pooled across all models; error bars = SE; dashed = human baseline)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, frameon=True)
    sns.despine()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: '{output_path}'")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder",     "-f", default="results")
    p.add_argument("--output-dir", "-o", default="plots")
    args = p.parse_args()

    print(f"Loading data from '{args.folder}'...")
    overall, foundation = load_summaries(args.folder)
    print(f"  Loaded {len(overall)} models.\n")

    plot_overall(overall,
                 os.path.join(args.output_dir, "overall_chart.png"))

    plot_within_lab(overall,
                    os.path.join(args.output_dir, "within_lab_contrasts.png"))

    plot_foundation_gaps(overall, foundation,
                         os.path.join(args.output_dir, "foundation_gaps.png"))


if __name__ == "__main__":
    main()
