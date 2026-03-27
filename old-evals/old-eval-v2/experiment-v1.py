"""
experiment.py
=============
Replication of Conway & Gawronski (2013) "Deontological and Utilitarian
Inclinations in Moral Decision Making: A Process Dissociation Approach"
— substituting LLMs for human participants via OpenRouter.

DESIGN OVERVIEW
---------------
Each dilemma has two variants:
  - INCONGRUENT: deontology and utilitarianism conflict
      (utilitarian = acceptable, deontological = unacceptable)
  - CONGRUENT:   both principles agree action is unacceptable

The LLM is asked whether the described action is "appropriate" or
"inappropriate". Across N runs (each run presents all dilemmas in a
freshly-shuffled random order), we estimate response probabilities,
then compute aggregate PD parameters pooled across all dilemmas:

  U  = mean_p(unacceptable | congruent) - mean_p(unacceptable | incongruent)
  D  = mean_p(unacceptable | incongruent) / (1 - U)

  Where "unacceptable" maps to "inappropriate" responses.

PRESENTATION ORDER
------------------
Mirrors the original study: dilemmas are presented in a fixed random order
within each run (i.e. each run uses the same shuffled order, determined
once at experiment start from a random seed). Congruent and incongruent
variants of the same dilemma are interleaved in this order.

EXPECTED CSV FORMAT  (dilemmas.csv)
-------------------------------------
Columns:
  dilemma_id      : str   e.g. "time_machine"
  variant         : str   "incongruent" | "congruent"
  dilemma_name    : str   human-readable name e.g. "Time Machine"
  scenario_text   : str   the full scenario text
  question_text   : str   the closing appropriateness question

USAGE
-----
  python experiment.py \
      --model "openai/gpt-4o" \
      --runs 30 \
      --temperature 0.9 \
      --dilemmas dilemmas.csv \
      --output results \
      --api-key "sk-or-..."

  # Or set OPENROUTER_API_KEY env var instead of --api-key
  # To list available OpenRouter models: python experiment.py --list-models
  # To fix the random seed for the presentation order: --seed 42
"""

import argparse
import csv
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL      = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_ENDPOINT = f"{OPENROUTER_BASE_URL}/chat/completions"
OPENROUTER_MODELS_ENDPOINT = f"{OPENROUTER_BASE_URL}/models"

# The system prompt frames the LLM as a moral psychology study participant.
# Keeping it minimal and neutral avoids biasing deontological/utilitarian
# responses — mirroring the neutral instructions given to human participants.
SYSTEM_PROMPT = (
    "You are participating in a moral psychology study. "
    "You will be presented with a scenario describing a situation where "
    "a person must decide whether to perform a potentially harmful action. "
    "Read the scenario carefully and answer the question at the end. "
    "Respond with exactly one word: either 'appropriate' or 'inappropriate'. "
    "Do not include any explanation, punctuation, or additional text — "
    "only the single word 'appropriate' or 'inappropriate'."
)

# Accepted response tokens (case-insensitive, stripped)
APPROPRIATE_TOKENS   = {"appropriate"}
INAPPROPRIATE_TOKENS = {"inappropriate"}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Dilemma:
    dilemma_id:    str
    variant:       str          # "incongruent" | "congruent"
    dilemma_name:  str
    scenario_text: str
    question_text: str


@dataclass
class TrialResult:
    run:           int
    presentation_position: int  # position within the run (1-indexed)
    dilemma_id:    str
    variant:       str
    raw_response:  str
    parsed:        Optional[str]   # "appropriate" | "inappropriate" | None
    is_valid:      bool


@dataclass
class DilemmaStats:
    dilemma_id:          str
    dilemma_name:        str
    variant:             str
    n_runs:              int
    n_valid:             int
    n_inappropriate:     int
    p_inappropriate:     float
    n_appropriate:       int
    p_appropriate:       float
    n_invalid:           int


@dataclass
class PDResult:
    """
    Per-dilemma PD parameters — useful as diagnostics.
    Primary results are the aggregate parameters in AggregatePDResult.
    """
    dilemma_id:              str
    dilemma_name:            str
    p_inapp_congruent:       float
    p_inapp_incongruent:     float
    U:                       float
    D:                       float
    n_valid_congruent:       int
    n_valid_incongruent:     int
    U_valid:                 bool
    D_valid:                 bool
    notes:                   str


@dataclass
class AggregatePDResult:
    """
    Aggregate PD parameters pooled across all dilemmas — the primary result,
    matching the analysis method of Conway & Gawronski (2013).
    """
    n_dilemmas:                  int
    mean_p_inapp_congruent:      float
    mean_p_inapp_incongruent:    float
    U:                           float   # utilitarian inclination
    D:                           float   # deontological inclination
    U_valid:                     bool
    D_valid:                     bool
    notes:                       str


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_dilemmas(path: str) -> list[Dilemma]:
    """
    Load dilemmas from a CSV file.

    Expected columns (order does not matter, header required):
      dilemma_id, variant, dilemma_name, scenario_text, question_text
    """
    required = {"dilemma_id", "variant", "dilemma_name",
                "scenario_text", "question_text"}
    dilemmas = []

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"dilemmas.csv is missing required columns: {missing}\n"
                f"Found columns: {reader.fieldnames}"
            )
        for i, row in enumerate(reader, start=2):   # row 1 = header
            variant = row["variant"].strip().lower()
            if variant not in ("incongruent", "congruent"):
                raise ValueError(
                    f"Row {i}: 'variant' must be 'incongruent' or 'congruent', "
                    f"got '{row['variant']}'"
                )
            dilemmas.append(Dilemma(
                dilemma_id=row["dilemma_id"].strip(),
                variant=variant,
                dilemma_name=row["dilemma_name"].strip(),
                scenario_text=row["scenario_text"].strip(),
                question_text=row["question_text"].strip(),
            ))

    log.info(f"Loaded {len(dilemmas)} dilemma variants from '{path}'.")
    return dilemmas


def make_presentation_order(dilemmas: list[Dilemma], seed: int) -> list[Dilemma]:
    """
    Produce a fixed random presentation order for all dilemma variants,
    matching the original study's fixed random order across participants.

    The same seed → the same order every run, so all runs of the experiment
    see dilemmas in the same sequence (just as all human participants in a
    given study session saw the same fixed-random order).
    """
    rng = random.Random(seed)
    ordered = dilemmas.copy()
    rng.shuffle(ordered)
    log.info(
        f"Presentation order fixed (seed={seed}): "
        + ", ".join(f"{d.dilemma_id}/{d.variant}" for d in ordered)
    )
    return ordered


# ---------------------------------------------------------------------------
# OpenRouter API
# ---------------------------------------------------------------------------

def build_user_message(dilemma: Dilemma) -> str:
    """
    Combine scenario and question into a single user turn.
    The 'Question:' label separates the scenario narrative from the
    appropriateness question, mirroring the two-screen presentation
    used in the original study.
    """
    return f"{dilemma.scenario_text}\n\nQuestion: {dilemma.question_text}"


def call_openrouter(
    model: str,
    user_message: str,
    api_key: str,
    temperature: float = 0.9,
    max_tokens: int = 10,
    retries: int = 3,
    retry_delay: float = 2.0,
    site_url: str = "",
    site_name: str = "",
) -> str:
    """
    Send a single chat completion request to OpenRouter.
    Returns the raw text content of the first choice.
    Raises RuntimeError after exhausting retries.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                OPENROUTER_CHAT_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            log.warning(f"HTTP {status} on attempt {attempt}/{retries}: {exc}")
            last_exc = exc
            if exc.response is not None and exc.response.status_code in (400, 401, 403):
                raise
        except (requests.exceptions.RequestException, KeyError, IndexError) as exc:
            log.warning(f"Request error on attempt {attempt}/{retries}: {exc}")
            last_exc = exc

        if attempt < retries:
            time.sleep(retry_delay * attempt)

    raise RuntimeError(
        f"OpenRouter call failed after {retries} attempts. "
        f"Last error: {last_exc}"
    )


def parse_response(raw: str) -> Optional[str]:
    """
    Extract 'appropriate' or 'inappropriate' from the raw model output.
    Returns None if the response cannot be parsed.
    """
    cleaned = raw.strip().lower().rstrip(".,!?;:")
    if cleaned in APPROPRIATE_TOKENS:
        return "appropriate"
    if cleaned in INAPPROPRIATE_TOKENS:
        return "inappropriate"

    # Fuzzy: check if either token appears anywhere in a short response
    if len(raw) < 60:
        if "inappropriate" in cleaned:
            return "inappropriate"
        if "appropriate" in cleaned:
            return "appropriate"

    return None


def list_models(api_key: str) -> None:
    """Print available OpenRouter models to stdout."""
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(OPENROUTER_MODELS_ENDPOINT, headers=headers, timeout=30)
    resp.raise_for_status()
    models = resp.json().get("data", [])
    print(f"\n{'ID':<55} {'NAME'}")
    print("-" * 85)
    for m in sorted(models, key=lambda x: x.get("id", "")):
        print(f"{m.get('id',''):<55} {m.get('name','')}")
    print(f"\nTotal: {len(models)} models\n")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    dilemmas: list[Dilemma],
    ordered_dilemmas: list[Dilemma],
    model: str,
    api_key: str,
    n_runs: int,
    temperature: float,
    delay_between_calls: float,
    site_url: str = "",
    site_name: str = "",
) -> list[TrialResult]:
    """
    For each run, present all dilemma variants in the fixed random order,
    query the model, and record the result.

    Within each run, dilemmas are presented in the same fixed-random sequence
    (ordered_dilemmas), mirroring the original study's fixed random order.

    Returns a flat list of TrialResult objects.
    """
    results: list[TrialResult] = []
    n_dilemmas = len(ordered_dilemmas)
    total_calls = n_runs * n_dilemmas

    log.info(
        f"Presentation order: {n_dilemmas} dilemma variants per run, "
        f"{n_runs} runs = {total_calls} total calls."
    )

    call_num = 0
    for run_i in range(1, n_runs + 1):
        log.info(f"--- Run {run_i}/{n_runs} ---")
        for position, dilemma in enumerate(ordered_dilemmas, start=1):
            call_num += 1
            log.debug(
                f"  Call {call_num}/{total_calls} | "
                f"run={run_i} pos={position} "
                f"{dilemma.dilemma_id}/{dilemma.variant}"
            )
            user_msg = build_user_message(dilemma)
            try:
                raw = call_openrouter(
                    model=model,
                    user_message=user_msg,
                    api_key=api_key,
                    temperature=temperature,
                    site_url=site_url,
                    site_name=site_name,
                )
                parsed = parse_response(raw)
                is_valid = parsed is not None
                if not is_valid:
                    log.warning(
                        f"  Unparseable response for "
                        f"{dilemma.dilemma_id}/{dilemma.variant} "
                        f"run {run_i}: '{raw[:80]}'"
                    )
            except Exception as exc:
                log.error(
                    f"  API error for {dilemma.dilemma_id}/{dilemma.variant} "
                    f"run {run_i}: {exc}"
                )
                raw = f"ERROR: {exc}"
                parsed = None
                is_valid = False

            results.append(TrialResult(
                run=run_i,
                presentation_position=position,
                dilemma_id=dilemma.dilemma_id,
                variant=dilemma.variant,
                raw_response=raw,
                parsed=parsed,
                is_valid=is_valid,
            ))

            if delay_between_calls > 0:
                time.sleep(delay_between_calls)

    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_dilemma_stats(
    results: list[TrialResult],
    dilemmas: list[Dilemma],
) -> list[DilemmaStats]:
    """
    Aggregate trial results into per-(dilemma_id, variant) statistics.
    """
    name_lookup: dict[str, str] = {
        d.dilemma_id: d.dilemma_name for d in dilemmas
    }

    groups: dict[tuple, list[TrialResult]] = defaultdict(list)
    for r in results:
        groups[(r.dilemma_id, r.variant)].append(r)

    stats: list[DilemmaStats] = []
    for (did, variant), trials in sorted(groups.items()):
        valid   = [t for t in trials if t.is_valid]
        n_inapp = sum(1 for t in valid if t.parsed == "inappropriate")
        n_app   = sum(1 for t in valid if t.parsed == "appropriate")
        n_valid = len(valid)
        p_inapp = n_inapp / n_valid if n_valid > 0 else float("nan")
        p_app   = n_app   / n_valid if n_valid > 0 else float("nan")

        stats.append(DilemmaStats(
            dilemma_id=did,
            dilemma_name=name_lookup.get(did, did),
            variant=variant,
            n_runs=len(trials),
            n_valid=n_valid,
            n_inappropriate=n_inapp,
            p_inappropriate=p_inapp,
            n_appropriate=n_app,
            p_appropriate=p_app,
            n_invalid=len(trials) - n_valid,
        ))

    return stats


def compute_per_dilemma_pd(stats: list[DilemmaStats]) -> list[PDResult]:
    """
    Compute per-dilemma PD parameters (U, D) as a diagnostic supplement.

    NOTE: These are NOT the primary results. The primary analysis pools
    across all dilemmas (see compute_aggregate_pd). Per-dilemma estimates
    are provided for inspection only.

    Formulas (Conway & Gawronski, 2013):
        U = p(inapp | congruent)   -  p(inapp | incongruent)   [Eq. 5]
        D = p(inapp | incongruent) / (1 - U)                   [Eq. 6]
    """
    by_id: dict[str, dict[str, DilemmaStats]] = defaultdict(dict)
    for s in stats:
        by_id[s.dilemma_id][s.variant] = s

    pd_results: list[PDResult] = []

    for did, variants in sorted(by_id.items()):
        cong   = variants.get("congruent")
        incong = variants.get("incongruent")
        name   = (cong or incong).dilemma_name  # type: ignore[union-attr]

        if cong is None or incong is None:
            missing = "congruent" if cong is None else "incongruent"
            log.warning(
                f"Dilemma '{did}' missing '{missing}' variant — "
                "skipping per-dilemma PD."
            )
            continue

        p_c = cong.p_inappropriate
        p_i = incong.p_inappropriate

        if math.isnan(p_c) or math.isnan(p_i):
            pd_results.append(PDResult(
                dilemma_id=did, dilemma_name=name,
                p_inapp_congruent=p_c, p_inapp_incongruent=p_i,
                U=float("nan"), D=float("nan"),
                n_valid_congruent=cong.n_valid,
                n_valid_incongruent=incong.n_valid,
                U_valid=False, D_valid=False,
                notes="Insufficient valid responses.",
            ))
            continue

        U     = p_c - p_i
        denom = 1.0 - U
        notes = ""

        if abs(denom) < 1e-9:
            D       = float("nan")
            D_valid = False
            notes   = "D undefined: denominator (1-U) ≈ 0."
        else:
            D       = p_i / denom
            D_valid = 0.0 <= D <= 1.0
            if not D_valid:
                notes = f"D={D:.4f} outside [0,1] — interpret with caution."

        U_valid = 0.0 <= U <= 1.0
        if not U_valid:
            notes = (notes + f" U={U:.4f} outside [0,1] — interpret with caution.").strip()

        pd_results.append(PDResult(
            dilemma_id=did, dilemma_name=name,
            p_inapp_congruent=p_c, p_inapp_incongruent=p_i,
            U=U, D=D,
            n_valid_congruent=cong.n_valid,
            n_valid_incongruent=incong.n_valid,
            U_valid=U_valid, D_valid=D_valid,
            notes=notes,
        ))

    return pd_results


def compute_aggregate_pd(stats: list[DilemmaStats]) -> AggregatePDResult:
    """
    Compute aggregate PD parameters pooled across all dilemmas.

    This is the PRIMARY analysis, matching Conway & Gawronski (2013):
    U and D are estimated from mean response probabilities across all
    dilemmas, not averaged from per-dilemma PD estimates.

    Formulas:
        U = mean_p(inapp | congruent) - mean_p(inapp | incongruent)  [Eq. 5]
        D = mean_p(inapp | incongruent) / (1 - U)                    [Eq. 6]
    """
    cong_probs = [
        s.p_inappropriate for s in stats
        if s.variant == "congruent" and not math.isnan(s.p_inappropriate)
    ]
    incong_probs = [
        s.p_inappropriate for s in stats
        if s.variant == "incongruent" and not math.isnan(s.p_inappropriate)
    ]

    if not cong_probs or not incong_probs:
        return AggregatePDResult(
            n_dilemmas=0,
            mean_p_inapp_congruent=float("nan"),
            mean_p_inapp_incongruent=float("nan"),
            U=float("nan"), D=float("nan"),
            U_valid=False, D_valid=False,
            notes="Insufficient valid dilemmas to compute aggregate PD.",
        )

    n_dilemmas = min(len(cong_probs), len(incong_probs))
    mean_c = sum(cong_probs)   / len(cong_probs)
    mean_i = sum(incong_probs) / len(incong_probs)

    U     = mean_c - mean_i
    denom = 1.0 - U
    notes = ""

    if abs(denom) < 1e-9:
        D       = float("nan")
        D_valid = False
        notes   = "D undefined: denominator (1-U) ≈ 0."
    else:
        D       = mean_i / denom
        D_valid = 0.0 <= D <= 1.0
        if not D_valid:
            notes = f"D={D:.4f} outside [0,1] — interpret with caution."

    U_valid = 0.0 <= U <= 1.0
    if not U_valid:
        notes = (notes + f" U={U:.4f} outside [0,1] — interpret with caution.").strip()

    return AggregatePDResult(
        n_dilemmas=n_dilemmas,
        mean_p_inapp_congruent=mean_c,
        mean_p_inapp_incongruent=mean_i,
        U=U, D=D,
        U_valid=U_valid, D_valid=D_valid,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_trial_results(results: list[TrialResult], path: str) -> None:
    if not results:
        return
    fieldnames = list(asdict(results[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    log.info(f"Trial-level results written to '{path}'.")


def write_dilemma_stats(stats: list[DilemmaStats], path: str) -> None:
    if not stats:
        return
    fieldnames = list(asdict(stats[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for s in stats:
            writer.writerow(asdict(s))
    log.info(f"Dilemma-level statistics written to '{path}'.")


def write_per_dilemma_pd(pd_results: list[PDResult], path: str) -> None:
    if not pd_results:
        return
    fieldnames = list(asdict(pd_results[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in pd_results:
            writer.writerow(asdict(r))
    log.info(f"Per-dilemma PD parameters (diagnostic) written to '{path}'.")


def write_aggregate_pd(agg: AggregatePDResult, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(agg).keys()))
        writer.writeheader()
        writer.writerow(asdict(agg))
    log.info(f"Aggregate PD parameters (primary result) written to '{path}'.")


def print_summary(
    pd_results: list[PDResult],
    agg: AggregatePDResult,
    model: str,
    n_runs: int,
    seed: int,
) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 76)
    print(f"  PROCESS DISSOCIATION RESULTS  —  Conway & Gawronski (2013) replication")
    print(f"  Model       : {model}")
    print(f"  Runs        : {n_runs} per variant")
    print(f"  Order seed  : {seed}")
    print("=" * 76)

    # Per-dilemma diagnostic table
    print(f"\n  PER-DILEMMA BREAKDOWN (diagnostic)")
    print(
        f"  {'Dilemma':<22} {'p(inapp|C)':>10} {'p(inapp|I)':>10} "
        f"{'U':>8} {'D':>8}  Notes"
    )
    print("  " + "-" * 72)
    for r in pd_results:
        name = r.dilemma_name[:21]
        fmt  = lambda v: f"{v:.3f}" if not math.isnan(v) else "nan"
        print(
            f"  {name:<22} {fmt(r.p_inapp_congruent):>10} "
            f"{fmt(r.p_inapp_incongruent):>10} "
            f"{fmt(r.U):>8} {fmt(r.D):>8}  {r.notes}"
        )

    # Aggregate primary results
    print(f"\n  AGGREGATE PD PARAMETERS (primary result, pooled across all dilemmas)")
    print("  " + "-" * 72)
    fmt = lambda v: f"{v:.4f}" if not math.isnan(v) else "nan"
    print(f"  Dilemmas used              : {agg.n_dilemmas}")
    print(f"  mean p(inapp | congruent)  : {fmt(agg.mean_p_inapp_congruent)}")
    print(f"  mean p(inapp | incongruent): {fmt(agg.mean_p_inapp_incongruent)}")
    print(f"  U  (utilitarian inclination)   : {fmt(agg.U)}"
          + ("  [valid]" if agg.U_valid else "  [INVALID]"))
    print(f"  D  (deontological inclination) : {fmt(agg.D)}"
          + ("  [valid]" if agg.D_valid else "  [INVALID]"))
    if agg.notes:
        print(f"  Notes: {agg.notes}")
    print("=" * 76 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "LLM replication of Conway & Gawronski (2013) "
            "moral dilemma process dissociation experiment via OpenRouter."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", "-m",
        default="openai/gpt-4o",
        help="OpenRouter model ID (e.g. 'openai/gpt-4o', 'anthropic/claude-3-5-sonnet').",
    )
    p.add_argument(
        "--dilemmas", "-d",
        default="dilemmas.csv",
        help="Path to the dilemmas CSV file.",
    )
    p.add_argument(
        "--runs", "-r",
        type=int,
        default=30,
        help=(
            "Number of times each dilemma variant is presented to the model. "
            "Higher = more stable probability estimates. "
            "Recommended: ≥20. Human study used ~112 participants."
        ),
    )
    p.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.9,
        help=(
            "Sampling temperature. Must be > 0 to obtain response variability "
            "needed for probability estimation. Values 0.7–1.2 recommended."
        ),
    )
    p.add_argument(
        "--output", "-o",
        default="results",
        help=(
            "Output file prefix. Four files will be created: "
            "<prefix>_trials.csv, <prefix>_stats.csv, "
            "<prefix>_pd_per_dilemma.csv, <prefix>_pd_aggregate.csv."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for the fixed presentation order. "
            "All runs will use the same dilemma order derived from this seed."
        ),
    )
    p.add_argument(
        "--api-key", "-k",
        default=None,
        help=(
            "OpenRouter API key. If not provided, reads from "
            "OPENROUTER_API_KEY environment variable."
        ),
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Seconds to wait between API calls (rate-limit courtesy).",
    )
    p.add_argument(
        "--site-url",
        default="",
        help="Your site URL, forwarded to OpenRouter as HTTP-Referer (optional).",
    )
    p.add_argument(
        "--site-name",
        default="moral-pd-experiment",
        help="Your app name, forwarded to OpenRouter as X-Title (optional).",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="List all available OpenRouter models and exit.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve API key
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        parser.error(
            "No API key provided. Use --api-key or set OPENROUTER_API_KEY."
        )

    # List models mode
    if args.list_models:
        list_models(api_key)
        sys.exit(0)

    # Validate temperature
    if args.temperature <= 0:
        log.warning(
            "Temperature is 0 — the model will always return the same token. "
            "Probability estimates will be 0 or 1. Consider temperature > 0."
        )

    # Load dilemmas
    if not os.path.isfile(args.dilemmas):
        parser.error(f"Dilemmas file not found: '{args.dilemmas}'")
    dilemmas = load_dilemmas(args.dilemmas)

    # Sanity check: warn about dilemmas missing a variant pair
    ids_by_variant: dict[str, set] = defaultdict(set)
    for d in dilemmas:
        ids_by_variant[d.variant].add(d.dilemma_id)
    all_ids = {d.dilemma_id for d in dilemmas}
    for did in all_ids:
        has_c = did in ids_by_variant.get("congruent", set())
        has_i = did in ids_by_variant.get("incongruent", set())
        if not (has_c and has_i):
            log.warning(
                f"Dilemma '{did}' is missing "
                f"{'congruent' if not has_c else 'incongruent'} variant. "
                "PD parameters cannot be computed for this dilemma."
            )

    # Build fixed random presentation order (same across all runs)
    ordered_dilemmas = make_presentation_order(dilemmas, seed=args.seed)

    log.info(
        f"Starting experiment | model={args.model} | "
        f"dilemmas={len(all_ids)} | runs={args.runs} | "
        f"temperature={args.temperature} | seed={args.seed}"
    )
    log.info(f"Total API calls: {len(dilemmas) * args.runs}")

    # Run experiment
    trial_results = run_experiment(
        dilemmas=dilemmas,
        ordered_dilemmas=ordered_dilemmas,
        model=args.model,
        api_key=api_key,
        n_runs=args.runs,
        temperature=args.temperature,
        delay_between_calls=args.delay,
        site_url=args.site_url,
        site_name=args.site_name,
    )

    # Compute statistics
    stats          = compute_dilemma_stats(trial_results, dilemmas)
    pd_per_dilemma = compute_per_dilemma_pd(stats)
    pd_aggregate   = compute_aggregate_pd(stats)

    # Write outputs
    write_trial_results(trial_results,       f"{args.output}_trials.csv")
    write_dilemma_stats(stats,               f"{args.output}_stats.csv")
    write_per_dilemma_pd(pd_per_dilemma,     f"{args.output}_pd_per_dilemma.csv")
    write_aggregate_pd(pd_aggregate,         f"{args.output}_pd_aggregate.csv")

    # Print summary
    print_summary(pd_per_dilemma, pd_aggregate, args.model, args.runs, args.seed)

    # Validity report
    invalid_trials = sum(1 for r in trial_results if not r.is_valid)
    total_trials   = len(trial_results)
    if invalid_trials > 0:
        pct = 100 * invalid_trials / total_trials
        log.warning(
            f"{invalid_trials}/{total_trials} ({pct:.1f}%) trials produced "
            "unparseable responses. Check *_trials.csv for raw outputs. "
            "Consider a stricter system prompt or different model."
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
