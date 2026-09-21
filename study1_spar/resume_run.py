"""
resume_run.py
=============
Resumes an interrupted experiment run by reading the existing *_trials.csv,
determining how many complete runs are already saved, then continuing from
where it left off and appending new trials to the same file.

Stats and summary CSVs are regenerated from all trials (existing + new) at
the end.

USAGE
-----
  python resume_run.py --trials results/deepseek-r1-n50_trials.csv --runs 50
  python resume_run.py --trials results/qwen3-235b-thinking-n50_trials.csv --runs 50 --max-tokens 3000 --timeout 300
"""

import argparse
import csv
import os
import sys
import time

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "experiment_v3",
    os.path.join(os.path.dirname(__file__), "experiment-v3.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Import everything we need from the main experiment script
run_experiment  = _mod.run_experiment
compute_stats   = _mod.compute_stats
compute_summary = _mod.compute_summary
append_csv      = _mod.append_csv
load_dilemmas   = _mod.load_dilemmas
make_order      = _mod.make_order
call_openrouter = _mod.call_openrouter
TRIAL_FIELDS    = _mod.TRIAL_FIELDS

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


def count_complete_runs(trials_path):
    """
    Read the trials CSV and return:
      - model id
      - number of fully-complete runs (all 80 dilemmas present)
      - the raw row dicts for reuse in stats
    A run is complete if all 80 dilemma positions appear for that run number.
    """
    with open(trials_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        return None, 0, []

    model = rows[0]["model"]

    # Count how many dilemmas exist per run number
    from collections import Counter
    run_counts = Counter(int(r["run"]) for r in rows)

    # A complete run has exactly 80 dilemma entries
    complete_runs = sorted(rn for rn, cnt in run_counts.items() if cnt >= 80)
    n_complete = len(complete_runs)

    return model, n_complete, rows


def resume(trials_path, total_runs, dilemmas_path, seed, temperature, delay,
           simple, max_tokens, timeout, api_key):

    base         = trials_path.replace("_trials.csv", "")
    stats_path   = f"{base}_stats.csv"
    summary_path = f"{base}_summary.csv"

    if not os.path.isfile(trials_path):
        log.error(f"Trials file not found: '{trials_path}'")
        sys.exit(1)

    model, n_complete, existing_rows = count_complete_runs(trials_path)
    log.info(f"Model: {model}")
    log.info(f"Found {n_complete}/{total_runs} complete runs in '{trials_path}'.")

    if n_complete >= total_runs:
        log.info("Run is already complete — nothing to do.")
        return

    runs_remaining = total_runs - n_complete
    log.info(f"Resuming from run {n_complete + 1} — {runs_remaining} runs remaining.")

    dilemmas = load_dilemmas(dilemmas_path)
    ordered  = make_order(dilemmas, seed)

    # Append new trials to the existing file
    with open(trials_path, "a", newline="", encoding="utf-8") as trial_fh:
        trial_writer = csv.DictWriter(trial_fh, fieldnames=TRIAL_FIELDS)
        # No header — we're appending

        # Monkey-patch run numbers so they continue from where we left off
        # We do this by wrapping run_experiment with an offset
        original_run_experiment = run_experiment

        def run_with_offset(ordered, model, api_key, n_runs, temperature, delay,
                            simple=False, max_tokens=150, timeout=120,
                            trial_writer=None, trial_fh=None):
            """Identical to run_experiment but offsets run numbers by n_complete."""
            import math, time as _time
            from dotenv import load_dotenv
            system_prompt = _mod.SYSTEM_PROMPT_SIMPLE if simple else _mod.SYSTEM_PROMPT
            trials    = []
            total     = len(ordered) * n_runs
            call_n    = 0
            run_start = _time.time()
            for run_i in range(1, n_runs + 1):
                actual_run = run_i + n_complete
                log.info(f"--- Run {actual_run}/{total_runs} ---")
                for pos, d in enumerate(ordered, 1):
                    call_n += 1
                    if call_n > 1:
                        elapsed = _time.time() - run_start
                        avg     = elapsed / (call_n - 1)
                        eta     = avg * (total - call_n + 1)
                        log.info(f"  Call {call_n}/{total} | ETA {_mod._fmt_duration(eta)} | {d['dilemma_id']}")
                    else:
                        log.info(f"  Call {call_n}/{total} | {d['dilemma_id']}")
                    try:
                        raw = call_openrouter(
                            model, _mod.build_prompt(d, simple), api_key,
                            temperature, system_prompt,
                            max_tokens=max_tokens, timeout=timeout
                        )
                        if simple:
                            endorsement = _mod.parse_response_simple(raw)
                            likelihood = confidence = None
                        else:
                            endorsement, likelihood, confidence = _mod.parse_response(raw)
                    except Exception as e:
                        log.error(f"  API error {d['dilemma_id']} run {actual_run}: {e}")
                        raw         = f"ERROR: {e}"
                        endorsement = likelihood = confidence = None

                    valid = (
                        endorsement is not None if simple
                        else all(x is not None for x in (endorsement, likelihood, confidence))
                    )
                    if not valid:
                        log.warning(f"  Incomplete response {d['dilemma_id']} run {actual_run}: '{raw[:100]}'")

                    row = {
                        "run":          actual_run,
                        "position":     pos,
                        "model":        model,
                        "dilemma_id":   d["dilemma_id"],
                        "foundation":   d["foundation"],
                        "domain":       d["domain"],
                        "variant":      d["variant"],
                        "raw_response": raw,
                        "endorsement":  endorsement,
                        "likelihood":   likelihood,
                        "confidence":   confidence,
                        "fully_valid":  valid,
                    }
                    trials.append(row)
                    if trial_writer:
                        trial_writer.writerow(row)
                        if trial_fh:
                            trial_fh.flush()
                    if delay > 0:
                        _time.sleep(delay)
            return trials

        new_trials = run_with_offset(
            ordered, model, api_key, runs_remaining, temperature, delay, simple,
            max_tokens=max_tokens, timeout=timeout,
            trial_writer=trial_writer, trial_fh=trial_fh,
        )

    log.info("New trials complete. Regenerating stats and summary from all trials...")

    # Reload all trials (existing + new) for clean stats
    with open(trials_path, newline="", encoding="utf-8") as fh:
        all_rows = list(csv.DictReader(fh))

    # Convert types
    for row in all_rows:
        row["run"]      = int(row["run"])
        row["position"] = int(row["position"])
        for field in ("likelihood", "confidence"):
            try:
                row[field] = int(row[field]) if row[field] not in ("", "None", None) else None
            except (ValueError, TypeError):
                row[field] = None
        row["fully_valid"] = row["fully_valid"] in (True, "True")

    stats   = compute_stats(all_rows)
    summary = compute_summary(stats, model, total_runs)

    # Overwrite stats/summary (regenerated from scratch)
    def write_csv(rows, path):
        if not rows: return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log.info(f"Written: '{path}'")

    write_csv(stats,   stats_path)
    write_csv(summary, summary_path)
    log.info("Done.")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--trials",      "-t", required=True,
                   help="Path to the existing *_trials.csv to resume.")
    p.add_argument("--runs",        "-r", type=int, required=True,
                   help="Total number of runs the experiment should have (e.g. 50).")
    p.add_argument("--dilemmas",    "-d",
                   default=os.path.join(os.path.dirname(__file__),
                                        "..", "moral_dilemmas_survey_revised.csv"),
                   help="Path to dilemmas CSV.")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--temperature", "-T", type=float, default=1.0)
    p.add_argument("--delay",             type=float, default=0.25)
    p.add_argument("--max-tokens",        type=int,   default=150)
    p.add_argument("--timeout",           type=int,   default=120)
    p.add_argument("--simple",      "-s", action="store_true")
    p.add_argument("--api-key",     "-k", default=None)
    args = p.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        p.error("No API key. Use --api-key or set OPENROUTER_API_KEY.")

    resume(
        trials_path   = args.trials,
        total_runs    = args.runs,
        dilemmas_path = args.dilemmas,
        seed          = args.seed,
        temperature   = args.temperature,
        delay         = args.delay,
        simple        = args.simple,
        max_tokens    = args.max_tokens,
        timeout       = args.timeout,
        api_key       = api_key,
    )


if __name__ == "__main__":
    main()
