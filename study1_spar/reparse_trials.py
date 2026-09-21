"""
reparse_trials.py
=================
Re-parses raw_response values in an existing *_trials.csv using the current
parse_response logic, then rewrites the trials file and regenerates
*_stats.csv and *_summary.csv.

Use this when a parser bug caused high invalid rates on an already-completed run.

USAGE
-----
  python reparse_trials.py --trials results/nova-premier-n50_trials.csv
"""

import argparse
import csv
import os
import sys

# Import parsing + stats functions from the main experiment script
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "experiment_v3",
    os.path.join(os.path.dirname(__file__), "experiment-v3.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_response  = _mod.parse_response
compute_stats   = _mod.compute_stats
compute_summary = _mod.compute_summary


def reparse(trials_path):
    base = trials_path.replace("_trials.csv", "")
    stats_path   = f"{base}_stats.csv"
    summary_path = f"{base}_summary.csv"

    # Read existing trials
    with open(trials_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print("No rows found.")
        return

    model   = rows[0]["model"]
    n_runs  = int(rows[0]["run"]) if rows else 0
    for r in rows:
        if int(r["run"]) > n_runs:
            n_runs = int(r["run"])

    print(f"Re-parsing {len(rows)} trials for {model} (n_runs={n_runs})...")

    reparsed = []
    improved = 0
    for row in rows:
        raw = row["raw_response"]
        if raw.startswith("ERROR:"):
            endorsement = likelihood = confidence = None
            valid = False
        else:
            endorsement, likelihood, confidence = parse_response(raw)
            valid = all(x is not None for x in (endorsement, likelihood, confidence))

        was_valid = row["fully_valid"] == "True"
        if valid and not was_valid:
            improved += 1

        row["endorsement"]  = endorsement
        row["likelihood"]   = likelihood
        row["confidence"]   = confidence
        row["fully_valid"]  = valid
        reparsed.append(row)

    print(f"  Recovered {improved} previously-invalid trials.")

    # Rewrite trials CSV
    with open(trials_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(reparsed[0].keys()))
        w.writeheader()
        w.writerows(reparsed)
    print(f"  Rewritten: '{trials_path}'")

    # Regenerate stats + summary
    # Convert numeric fields back from strings
    for row in reparsed:
        row["run"]      = int(row["run"])
        row["position"] = int(row["position"])
        for field in ("likelihood", "confidence"):
            try:
                row[field] = int(row[field]) if row[field] not in ("", "None", None) else None
            except (ValueError, TypeError):
                row[field] = None
        row["fully_valid"] = row["fully_valid"] in (True, "True")

    stats   = compute_stats(reparsed)
    summary = compute_summary(stats, model, n_runs)

    # Overwrite (not append) since we're regenerating from scratch
    def write_csv(rows, path):
        if not rows: return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  Rewritten: '{path}'")

    write_csv(stats,   stats_path)
    write_csv(summary, summary_path)
    print("Done.")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--trials", "-t", required=True,
                   help="Path to the *_trials.csv file to re-parse.")
    args = p.parse_args()

    if not os.path.isfile(args.trials):
        p.error(f"File not found: '{args.trials}'")

    reparse(args.trials)


if __name__ == "__main__":
    main()
