"""
plot_foundations.py
===================
Creates a 2×3 faceted bar chart showing harm endorsement (congruent vs incongruent)
broken down by each of the 5 moral foundations + an overall panel.

One panel per foundation, models on the x-axis, sorted by overall gap.
Human baseline shown as dotted reference lines.

USAGE
-----
  python plot_foundations.py
  python plot_foundations.py --folder results --output plots/foundation_bars.png
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
# Config — keep in sync with plot_results.py
# ---------------------------------------------------------------------------
MODEL_LABELS = {
    "openai/gpt-4o":                        "GPT-4o",
    "openai/gpt-4o-mini":                   "GPT-4o-mini",
    "openai/gpt-5":                         "GPT-5",
    "openai/gpt-5-chat":                    "GPT-5",
    "openai/o1":                            "o1",
    "openai/o3":                            "o3",
    "anthropic/claude-3.5-sonnet":          "Claude 3.5 Sonnet",
    "anthropic/claude-opus-4":              "Claude Opus 4",
    "google/gemini-2.5-pro":               "Gemini 2.5 Pro",
    "google/gemma-3-27b-it":               "Gemma 3 27B",
    "x-ai/grok-3":                         "Grok 3",
    "x-ai/grok-4":                         "Grok 4",
    "amazon/nova-premier-v1":              "Nova Premier",
    "microsoft/phi-4":                     "Phi-4",
    "meta-llama/llama-3.3-70b-instruct":   "Llama 3.3 70B",
    "meta-llama/llama-4-maverick":         "Llama 4 Maverick",
    "mistralai/mixtral-8x22b-instruct":    "Mixtral 8x22B",
    "mistralai/mistral-large-2512":        "Mistral Large 3",
    "deepseek/deepseek-r1":               "DeepSeek R1",
    "deepseek/deepseek-r1-0528":          "DeepSeek R1-0528",
    "deepseek/deepseek-chat-v3-0324":     "DeepSeek V3",
    "qwen/qwen3-235b-a22b":               "Qwen3 235B",
    "qwen/qwen3-235b-a22b-thinking-2507": "Qwen3 235B (Think)",
    "z-ai/glm-4.7":                       "GLM-4.7",
    "baidu/ernie-4.5-300b-a47b":          "Ernie 4.5",
    "upstage/solar-pro-3":                "Solar Pro 3",
    "cohere/command-a":                   "Command A",
    "openai/gpt-3.5-turbo":              "GPT-3.5 Turbo",
    "meta-llama/llama-3.2-3b-instruct":  "Llama 3.2 3B",
    "meta-llama/llama-3.1-8b-instruct":  "Llama 3.1 8B",
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

HUMAN = {
    "authority": {"cong": 25.38, "incong": 56.08},   # overall used as proxy
    "care":      {"cong": 25.38, "incong": 56.08},
    "fairness":  {"cong": 25.38, "incong": 56.08},
    "loyalty":   {"cong": 25.38, "incong": 56.08},
    "purity":    {"cong": 25.38, "incong": 56.08},
    "overall":   {"cong": 25.38, "incong": 56.08},
}

FOUNDATIONS = ["authority", "care", "fairness", "loyalty", "purity"]
FOUNDATION_TITLES = {
    "authority": "Authority",
    "care":      "Care",
    "fairness":  "Fairness",
    "loyalty":   "Loyalty",
    "purity":    "Purity",
    "overall":   "Overall",
}

COLOR_CONGRUENT   = "#E8756A"
COLOR_INCONGRUENT = "#63C6C4"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _f(v):
    try:
        x = float(v)
        return float("nan") if math.isnan(x) else x
    except (ValueError, TypeError):
        return float("nan")


def load_data(folder):
    """Returns:
      overall_order : list of model_ids sorted by overall gap desc
      data          : {model_id: {foundation: {cong_mean, cong_se, incong_mean, incong_se, label, reasoning}}}
    """
    pattern = os.path.join(folder, "*_summary.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No *_summary.csv files in '{folder}'.")

    raw = {}   # model_id -> foundation -> row

    for fpath in files:
        with open(fpath, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                model = row["model"]
                level = row.get("level", "overall")
                found = row.get("foundation", "all")
                key   = "overall" if (level == "overall" or found == "all") else found

                entry = {
                    "cong_mean":   _f(row["mean_harm_endorsement_congruent"]),
                    "cong_se":     _f(row["se_harm_endorsement_congruent"]),
                    "incong_mean": _f(row["mean_harm_endorsement_incongruent"]),
                    "incong_se":   _f(row["se_harm_endorsement_incongruent"]),
                    "label":       MODEL_LABELS.get(model, model),
                    "reasoning":   model in REASONING_MODELS,
                }
                raw.setdefault(model, {})[key] = entry

    # Sort models by overall gap descending
    def gap(m):
        ov = raw[m].get("overall", {})
        c  = ov.get("cong_mean",   float("nan"))
        i  = ov.get("incong_mean", float("nan"))
        return (i - c) if not (math.isnan(c) or math.isnan(i)) else -999

    overall_order = sorted(raw.keys(), key=gap, reverse=True)
    return overall_order, raw


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_foundations(overall_order, data, output_path):
    panels = FOUNDATIONS + ["overall"]   # 6 panels in a 2×3 grid

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes_flat = axes.flatten()

    n      = len(overall_order)
    x      = np.arange(n)
    width  = 0.35
    offset = width / 2

    for ax, foundation in zip(axes_flat, panels):
        cong_means, cong_ses     = [], []
        incong_means, incong_ses = [], []
        labels                   = []
        is_reasoning             = []

        for model in overall_order:
            fd = data[model].get(foundation, {})
            cong_means.append(fd.get("cong_mean",   float("nan")))
            cong_ses.append(  fd.get("cong_se",     0))
            incong_means.append(fd.get("incong_mean", float("nan")))
            incong_ses.append(  fd.get("incong_se",   0))
            labels.append(data[model].get("overall", data[model].get(foundation, {})).get("label",
                          MODEL_LABELS.get(model, model)))
            is_reasoning.append(data[model].get("overall", {}).get("reasoning", False)
                                or data[model].get(foundation, {}).get("reasoning", False))

        # Replace nan SE with 0 for error bars
        cong_ses_plot   = [0 if math.isnan(v) else v for v in cong_ses]
        incong_ses_plot = [0 if math.isnan(v) else v for v in incong_ses]

        ax.bar(x - offset, cong_means,   width, color=COLOR_CONGRUENT,
               yerr=cong_ses_plot,   capsize=3, zorder=3,
               error_kw={"elinewidth": 0.8, "ecolor": "black"})
        ax.bar(x + offset, incong_means, width, color=COLOR_INCONGRUENT,
               yerr=incong_ses_plot, capsize=3, zorder=3,
               error_kw={"elinewidth": 0.8, "ecolor": "black"})

        # Human baseline dotted lines
        h = HUMAN[foundation]
        ax.axhline(h["cong"],   color=COLOR_CONGRUENT,   linestyle="--",
                   linewidth=1.2, alpha=0.7, zorder=2)
        ax.axhline(h["incong"], color=COLOR_INCONGRUENT, linestyle="--",
                   linewidth=1.2, alpha=0.7, zorder=2)

        tick_labels = [f"{lbl} *" if r else lbl
                       for lbl, r in zip(labels, is_reasoning)]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=8.5)
        ax.set_title(FOUNDATION_TITLES[foundation], fontsize=12, fontweight="bold", pad=6)
        ax.set_ylim(0, 112)
        ax.set_ylabel("% Harm Endorsement" if ax in axes[:, 0] else "", fontsize=9)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)

    # Shared legend
    legend_handles = [
        mpatches.Patch(color=COLOR_CONGRUENT,   label="Congruent (no utilitarian justification)"),
        mpatches.Patch(color=COLOR_INCONGRUENT, label="Incongruent (utilitarian justification)"),
        plt.Line2D([0], [0], color=COLOR_CONGRUENT,   linestyle="--", linewidth=1.5,
                   label="Human baseline — congruent (C&G 2013)"),
        plt.Line2D([0], [0], color=COLOR_INCONGRUENT, linestyle="--", linewidth=1.5,
                   label="Human baseline — incongruent (C&G 2013)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=9.5, frameon=True, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Moral Dilemma Harm Endorsement by Foundation\n"
        "(sorted by overall gap; * = reasoning model; dotted lines = human baseline)",
        fontsize=13, fontweight="bold", y=1.01,
    )

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
    p.add_argument("--folder", "-f", default="results",
                   help="Folder containing *_summary.csv files.")
    p.add_argument("--output", "-o", default="plots/foundation_bars.png",
                   help="Output PNG path.")
    args = p.parse_args()

    overall_order, data = load_data(args.folder)
    print(f"Loaded {len(overall_order)} model(s): "
          f"{[data[m].get('overall', data[m].get(list(data[m])[0], {})).get('label', m) for m in overall_order]}")
    plot_foundations(overall_order, data, args.output)


if __name__ == "__main__":
    main()
