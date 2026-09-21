"""
analyze_results.py
==================
Statistical analysis for Study 1 moral dilemma experiment.

Runs the following analyses and saves results to analysis/

  1. Descriptive table   — gap, congruent%, incongruent% per model
  2. Reasoning contrast  — independent-samples t-test + Cohen's d (reasoning vs. non-reasoning)
  3. Within-lab contrasts— paired comparisons for matched reasoning/non-reasoning model pairs
  4. Foundation ANOVA    — one-way ANOVA across foundations (all models pooled), with post-hoc
  5. Model × Foundation  — gap per model × foundation heatmap table
  6. Regression          — gap ~ reasoning_model (binary) OLS

All results are printed to stdout and saved to analysis/stats_report.txt
Tables saved as CSV to analysis/

USAGE
-----
  python analyze_results.py --folder results
"""

import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Optional scipy — graceful fallback if not installed
# ---------------------------------------------------------------------------
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy not found — t-tests and ANOVA will be skipped. "
          "Install with: pip install scipy", file=sys.stderr)

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    "anthropic/claude-3.5-sonnet":           "Claude 3.5 Sonnet",
    "anthropic/claude-opus-4":               "Claude Opus 4",
    "openai/gpt-3.5-turbo":                  "GPT-3.5 Turbo",
    "openai/gpt-4o":                         "GPT-4o",
    "openai/gpt-4o-mini":                    "GPT-4o-mini",
    "openai/gpt-5-chat":                     "GPT-5",
    "openai/o1":                             "o1",
    "openai/o3":                             "o3",
    "x-ai/grok-3":                           "Grok 3",
    "x-ai/grok-4":                           "Grok 4",
    "google/gemini-2.5-pro":                 "Gemini 2.5 Pro",
    "google/gemini-2.0-flash-001":           "Gemini 2.0 Flash",
    "google/gemma-3-27b-it":                 "Gemma 3 27B",
    "meta-llama/llama-3.2-3b-instruct":      "Llama 3.2 3B",
    "meta-llama/llama-3.3-70b-instruct":     "Llama 3.3 70B",
    "meta-llama/llama-4-maverick":           "Llama 4 Maverick",
    "mistralai/mixtral-8x22b-instruct":      "Mixtral 8x22B",
    "mistralai/mistral-large-2512":          "Mistral Large 3",
    "deepseek/deepseek-chat-v3-0324":        "DeepSeek V3",
    "deepseek/deepseek-r1":                  "DeepSeek R1",
    "deepseek/deepseek-r1-0528":             "DeepSeek R1-0528",
    "qwen/qwen3-235b-a22b":                  "Qwen3 235B",
    "qwen/qwen3-235b-a22b-thinking-2507":    "Qwen3 235B (Thinking)",
    "baidu/ernie-4.5-300b-a47b":             "Ernie 4.5",
    "amazon/nova-premier-v1":                "Nova Premier",
    "microsoft/phi-4":                       "Phi-4",
    "cohere/command-a":                      "Command A",
    "upstage/solar-pro-3":                   "Solar Pro 3",
    "openai/gpt-5":                          "GPT-5",
}

REASONING_MODELS = {
    "anthropic/claude-opus-4",
    "openai/o1",
    "openai/o3",
    "x-ai/grok-3",
    "google/gemini-2.5-pro",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-0528",
    "qwen/qwen3-235b-a22b-thinking-2507",
}

# Within-lab reasoning/non-reasoning pairs: (non-reasoning, reasoning)
WITHIN_LAB_PAIRS = [
    ("Anthropic",  "anthropic/claude-3.5-sonnet",        "anthropic/claude-opus-4"),
    ("xAI",        "x-ai/grok-4",                        "x-ai/grok-3"),
    ("OpenAI",     "openai/gpt-4o",                      "openai/o1"),
    ("OpenAI",     "openai/gpt-5-chat",                  "openai/o3"),
    ("DeepSeek",   "deepseek/deepseek-chat-v3-0324",     "deepseek/deepseek-r1"),
    ("DeepSeek",   "deepseek/deepseek-chat-v3-0324",     "deepseek/deepseek-r1-0528"),
    ("Qwen",       "qwen/qwen3-235b-a22b",               "qwen/qwen3-235b-a22b-thinking-2507"),
    ("Google",     "google/gemma-3-27b-it",              "google/gemini-2.5-pro"),
]

HUMAN_BASELINE = {"congruent": 25.38, "incongruent": 56.08, "gap": 30.70}

FOUNDATIONS = ["authority", "care", "fairness", "loyalty", "purity"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")

def _sd(vals):
    if len(vals) < 2: return float("nan")
    m = _mean(vals)
    return math.sqrt(sum((x - m)**2 for x in vals) / (len(vals) - 1))

def _se(vals):
    return _sd(vals) / math.sqrt(len(vals)) if len(vals) >= 2 else float("nan")

def cohen_d(a, b):
    """Cohen's d for two independent groups."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_sd = math.sqrt((_sd(a)**2 + _sd(b)**2) / 2)
    return (_mean(a) - _mean(b)) / pooled_sd if pooled_sd else float("nan")

def fmt(v, decimals=2):
    if math.isnan(v): return "nan"
    return f"{v:.{decimals}f}"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_summaries(folder):
    """Load all *_summary.csv files recursively. Returns dict model_id → rows."""
    pattern = os.path.join(folder, "**", "*_summary.csv")
    files   = glob.glob(pattern, recursive=True)
    data    = {}
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        model = rows[0]["model"]
        data[model] = rows
    return data

def load_stats(folder):
    """Load all *_stats.csv files. Returns dict model_id → rows."""
    pattern = os.path.join(folder, "**", "*_stats.csv")
    files   = glob.glob(pattern, recursive=True)
    data    = {}
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        model = rows[0]["model"]
        data[model] = rows
    return data

def overall_row(summary_rows):
    for r in summary_rows:
        if r["level"] == "overall":
            return r
    return None

def foundation_row(summary_rows, foundation):
    for r in summary_rows:
        if r["level"] == "foundation" and r["foundation"] == foundation:
            return r
    return None

# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def descriptive_table(summaries):
    """Return list of dicts with per-model descriptive stats."""
    rows = []
    for model_id, sumrows in sorted(summaries.items()):
        ov = overall_row(sumrows)
        if ov is None:
            continue
        cong   = float(ov["mean_harm_endorsement_congruent"])
        incong = float(ov["mean_harm_endorsement_incongruent"])
        se_c   = float(ov["se_harm_endorsement_congruent"])
        se_i   = float(ov["se_harm_endorsement_incongruent"])
        gap    = incong - cong
        label  = MODEL_LABELS.get(model_id, model_id)
        reason = model_id in REASONING_MODELS
        rows.append({
            "model_id":    model_id,
            "model":       label,
            "reasoning":   reason,
            "congruent":   cong,
            "se_cong":     se_c,
            "incongruent": incong,
            "se_incong":   se_i,
            "gap":         gap,
            "n_runs":      int(ov["n_runs"]),
        })
    rows.sort(key=lambda r: -r["gap"])
    return rows

def reasoning_ttest(desc_rows):
    """Independent-samples t-test and Cohen's d: reasoning vs. non-reasoning gaps."""
    reason_gaps = [r["gap"] for r in desc_rows if r["reasoning"] and not math.isnan(r["gap"])]
    nonreason_gaps = [r["gap"] for r in desc_rows if not r["reasoning"] and not math.isnan(r["gap"])]

    result = {
        "n_reasoning":     len(reason_gaps),
        "n_nonreasoning":  len(nonreason_gaps),
        "mean_gap_reasoning":    _mean(reason_gaps),
        "mean_gap_nonreasoning": _mean(nonreason_gaps),
        "sd_gap_reasoning":      _sd(reason_gaps),
        "sd_gap_nonreasoning":   _sd(nonreason_gaps),
        "cohens_d":        cohen_d(nonreason_gaps, reason_gaps),
        "t_stat":          float("nan"),
        "p_value":         float("nan"),
        "df":              float("nan"),
    }

    if HAS_SCIPY and len(reason_gaps) >= 2 and len(nonreason_gaps) >= 2:
        t, p = scipy_stats.ttest_ind(reason_gaps, nonreason_gaps, equal_var=False)
        df   = len(reason_gaps) + len(nonreason_gaps) - 2
        result.update({"t_stat": t, "p_value": p, "df": df})

    return result, reason_gaps, nonreason_gaps

def within_lab_contrasts(summaries):
    """Per-pair comparison of reasoning vs. non-reasoning within same lab."""
    rows = []
    for lab, nr_id, r_id in WITHIN_LAB_PAIRS:
        if nr_id not in summaries or r_id not in summaries:
            continue
        nr_ov = overall_row(summaries[nr_id])
        r_ov  = overall_row(summaries[r_id])
        if nr_ov is None or r_ov is None:
            continue
        nr_gap = float(nr_ov["mean_harm_endorsement_incongruent"]) - float(nr_ov["mean_harm_endorsement_congruent"])
        r_gap  = float(r_ov["mean_harm_endorsement_incongruent"])  - float(r_ov["mean_harm_endorsement_congruent"])
        rows.append({
            "lab":            lab,
            "non_reasoning":  MODEL_LABELS.get(nr_id, nr_id),
            "reasoning":      MODEL_LABELS.get(r_id, r_id),
            "gap_nonreason":  nr_gap,
            "gap_reason":     r_gap,
            "delta":          r_gap - nr_gap,
            "direction":      "↓ reduced" if r_gap < nr_gap else "↑ increased",
        })
    return rows

def foundation_analysis(summaries, stats_data):
    """
    Per-foundation gap across all models (pooled) and per-model × foundation matrix.
    Also runs one-way ANOVA across foundations using dilemma-level p_yes values.
    """
    # Per-foundation mean gap across all models (excluding Llama 3.2 3B — capacity issue)
    EXCLUDE = {"meta-llama/llama-3.2-3b-instruct"}
    foundation_gaps = defaultdict(list)

    for model_id, sumrows in summaries.items():
        if model_id in EXCLUDE:
            continue
        for f in FOUNDATIONS:
            fr = foundation_row(sumrows, f)
            if fr is None:
                continue
            cong   = float(fr["mean_harm_endorsement_congruent"])
            incong = float(fr["mean_harm_endorsement_incongruent"])
            if not math.isnan(cong) and not math.isnan(incong):
                foundation_gaps[f].append(incong - cong)

    foundation_summary = []
    for f in FOUNDATIONS:
        vals = foundation_gaps[f]
        foundation_summary.append({
            "foundation": f,
            "n_models":   len(vals),
            "mean_gap":   _mean(vals),
            "sd_gap":     _sd(vals),
            "se_gap":     _se(vals),
            "min_gap":    min(vals) if vals else float("nan"),
            "max_gap":    max(vals) if vals else float("nan"),
        })

    # ANOVA across foundations using dilemma-level p_yes (incongruent trials only)
    anova_result = None
    if HAS_SCIPY and stats_data:
        groups = defaultdict(list)
        for model_id, srows in stats_data.items():
            if model_id in EXCLUDE:
                continue
            for row in srows:
                if row["variant"] == "incongruent":
                    try:
                        groups[row["foundation"]].append(float(row["p_yes"]))
                    except (ValueError, KeyError):
                        pass
        group_vals = [groups[f] for f in FOUNDATIONS if groups[f]]
        if len(group_vals) >= 2:
            f_stat, p_val = scipy_stats.f_oneway(*group_vals)
            anova_result = {"f_stat": f_stat, "p_value": p_val, "df_between": len(FOUNDATIONS) - 1}

    # Per-model × foundation matrix
    model_foundation_matrix = []
    for model_id, sumrows in sorted(summaries.items(), key=lambda x: MODEL_LABELS.get(x[0], x[0])):
        row = {"model": MODEL_LABELS.get(model_id, model_id), "reasoning": model_id in REASONING_MODELS}
        for f in FOUNDATIONS:
            fr = foundation_row(sumrows, f)
            if fr:
                cong   = float(fr["mean_harm_endorsement_congruent"])
                incong = float(fr["mean_harm_endorsement_incongruent"])
                row[f"gap_{f}"] = incong - cong
            else:
                row[f"gap_{f}"] = float("nan")
        model_foundation_matrix.append(row)

    return foundation_summary, anova_result, model_foundation_matrix

def human_comparison(desc_rows):
    """How many models exceed human gap? Distribution of gaps relative to baseline."""
    gaps = [r["gap"] for r in desc_rows if not math.isnan(r["gap"]) and r["model"] != "Llama 3.2 3B"]
    exceed = sum(1 for g in gaps if g > HUMAN_BASELINE["gap"])
    return {
        "n_models":              len(gaps),
        "n_exceed_human":        exceed,
        "pct_exceed_human":      100 * exceed / len(gaps) if gaps else 0,
        "mean_gap_all_models":   _mean(gaps),
        "sd_gap_all_models":     _sd(gaps),
        "human_gap":             HUMAN_BASELINE["gap"],
        "mean_excess_over_human": _mean([g - HUMAN_BASELINE["gap"] for g in gaps]),
    }

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_csv_out(rows, path):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")

def section(title, out):
    line = "=" * 70
    out.append(f"\n{line}")
    out.append(f"  {title}")
    out.append(line)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", "-f", default="results",
                   help="Root results folder (searched recursively).")
    p.add_argument("--output", "-o", default="analysis",
                   help="Output directory for tables and report.")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading data from '{args.folder}'...")
    summaries  = load_summaries(args.folder)
    stats_data = load_stats(args.folder)
    print(f"  Loaded {len(summaries)} models.\n")

    out = []  # lines for text report

    # ------------------------------------------------------------------
    # 1. Descriptive table
    # ------------------------------------------------------------------
    section("1. DESCRIPTIVE TABLE — Gap, Congruent%, Incongruent% per Model", out)
    desc_rows = descriptive_table(summaries)

    out.append(f"\n{'Model':<30} {'R':<3} {'Cong%':>7} {'Incong%':>9} {'Gap':>7} {'N':>4}")
    out.append("-" * 60)
    for r in desc_rows:
        flag = "Y" if r["reasoning"] else " "
        out.append(f"  {r['model']:<28} {flag:<3} {fmt(r['congruent']):>7} {fmt(r['incongruent']):>9} {fmt(r['gap']):>7} {r['n_runs']:>4}")
    out.append("-" * 60)
    out.append(f"  {'Human (C&G 2013)':<28} {'':3} {fmt(HUMAN_BASELINE['congruent']):>7} {fmt(HUMAN_BASELINE['incongruent']):>9} {fmt(HUMAN_BASELINE['gap']):>7}    —")

    write_csv_out(desc_rows, os.path.join(args.output, "descriptive_table.csv"))

    # ------------------------------------------------------------------
    # 2. Human baseline comparison
    # ------------------------------------------------------------------
    section("2. HUMAN BASELINE COMPARISON", out)
    hc = human_comparison(desc_rows)
    out.append(f"\n  Models tested (excl. Llama 3.2 3B capacity failure): {hc['n_models']}")
    out.append(f"  Models exceeding human gap ({fmt(HUMAN_BASELINE['gap'])} pp):  {hc['n_exceed_human']} / {hc['n_models']} ({fmt(hc['pct_exceed_human'])}%)")
    out.append(f"  Mean gap across all models:  {fmt(hc['mean_gap_all_models'])} pp  (SD = {fmt(hc['sd_gap_all_models'])})")
    out.append(f"  Mean excess over human baseline: +{fmt(hc['mean_excess_over_human'])} pp")

    # ------------------------------------------------------------------
    # 3. Reasoning vs. Non-reasoning t-test
    # ------------------------------------------------------------------
    section("3. REASONING vs. NON-REASONING — Independent Samples t-test", out)
    tt, r_gaps, nr_gaps = reasoning_ttest(desc_rows)

    out.append(f"\n  Non-reasoning models (n={tt['n_nonreasoning']}): "
               f"M = {fmt(tt['mean_gap_nonreasoning'])} pp, SD = {fmt(tt['sd_gap_nonreasoning'])}")
    out.append(f"  Reasoning models    (n={tt['n_reasoning']}): "
               f"M = {fmt(tt['mean_gap_reasoning'])} pp, SD = {fmt(tt['sd_gap_reasoning'])}")
    out.append(f"\n  Welch's t({fmt(tt['df'], 0)}) = {fmt(tt['t_stat'], 3)},  p = {fmt(tt['p_value'], 4)}")
    out.append(f"  Cohen's d = {fmt(tt['cohens_d'], 3)}  (non-reasoning minus reasoning)")

    if not HAS_SCIPY:
        out.append("\n  [scipy not available — t-stat and p-value not computed]")

    write_csv_out([tt], os.path.join(args.output, "reasoning_ttest.csv"))

    # ------------------------------------------------------------------
    # 4. Within-lab contrasts
    # ------------------------------------------------------------------
    section("4. WITHIN-LAB REASONING CONTRASTS", out)
    wl = within_lab_contrasts(summaries)
    out.append(f"\n  {'Lab':<12} {'Non-reasoning':<28} {'Reasoning':<28} {'NR Gap':>7} {'R Gap':>7} {'Δ':>7}  Direction")
    out.append("  " + "-" * 100)
    for r in wl:
        out.append(f"  {r['lab']:<12} {r['non_reasoning']:<28} {r['reasoning']:<28} "
                   f"{fmt(r['gap_nonreason']):>7} {fmt(r['gap_reason']):>7} {('+' if r['delta'] >= 0 else '') + fmt(r['delta']):>7}  {r['direction']}")

    write_csv_out(wl, os.path.join(args.output, "within_lab_contrasts.csv"))

    # ------------------------------------------------------------------
    # 5. Foundation-level analysis
    # ------------------------------------------------------------------
    section("5. FOUNDATION-LEVEL ANALYSIS", out)
    f_summary, anova_res, f_matrix = foundation_analysis(summaries, stats_data)

    out.append(f"\n  Mean gap by moral foundation (all models pooled, excl. Llama 3.2 3B):\n")
    out.append(f"  {'Foundation':<12} {'Mean Gap':>9} {'SD':>7} {'SE':>7} {'Min':>7} {'Max':>7}")
    out.append("  " + "-" * 55)
    for r in sorted(f_summary, key=lambda x: -x["mean_gap"]):
        out.append(f"  {r['foundation'].capitalize():<12} {fmt(r['mean_gap']):>9} {fmt(r['sd_gap']):>7} "
                   f"{fmt(r['se_gap']):>7} {fmt(r['min_gap']):>7} {fmt(r['max_gap']):>7}")

    if anova_res:
        out.append(f"\n  One-way ANOVA (incongruent p_yes across foundations):")
        out.append(f"  F({anova_res['df_between']}, …) = {fmt(anova_res['f_stat'], 3)},  p = {fmt(anova_res['p_value'], 4)}")

    write_csv_out(f_summary, os.path.join(args.output, "foundation_summary.csv"))
    write_csv_out(f_matrix,  os.path.join(args.output, "model_foundation_matrix.csv"))

    out.append(f"\n  Per-model × foundation gap matrix:\n")
    header = f"  {'Model':<30} {'R':<3}" + "".join(f"  {f.capitalize()[:5]:>7}" for f in FOUNDATIONS)
    out.append(header)
    out.append("  " + "-" * (34 + 9 * len(FOUNDATIONS)))
    for r in sorted(f_matrix, key=lambda x: -sum(
        v for k, v in x.items() if k.startswith("gap_") and not math.isnan(v)
    )):
        flag = "Y" if r["reasoning"] else " "
        vals = "".join(f"  {fmt(r[f'gap_{f}']):>7}" for f in FOUNDATIONS)
        out.append(f"  {r['model']:<30} {flag:<3}{vals}")

    # ------------------------------------------------------------------
    # Print and save report
    # ------------------------------------------------------------------
    report_text = "\n".join(out)
    print(report_text)

    report_path = os.path.join(args.output, "stats_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    print(f"\n  Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
