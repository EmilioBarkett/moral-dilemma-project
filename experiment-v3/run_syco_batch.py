"""
Sequential sycophancy batch runner.
Runs models one at a time, updates llm_eval_list_v2.csv after each,
and logs progress to run_syco_batch.log.
"""

import subprocess, csv, sys, logging, time
from datetime import datetime
from pathlib import Path

# ── Models to run (cheap/fast → expensive/slow) ──────────────────────────────
MODELS = [
    ("google/gemma-3-27b-it",                  "gemma3-27b-n10"),
    ("meta-llama/llama-3.3-70b-instruct",       "llama33-70b-n10"),
    ("deepseek/deepseek-chat-v3-0324",          "deepseek-v3-n10"),
    ("mistralai/mixtral-8x22b-instruct",        "mixtral-8x22b-n10"),
    ("mistralai/mistral-large-2512",            "mistral-large3-n10"),
    ("cohere/command-a",                        "command-a-n10"),
    ("upstage/solar-pro-3",                     "solar-pro3-n10"),
    ("amazon/nova-premier-v1",                  "nova-premier-n10"),
    ("baidu/ernie-4.5-300b-a47b",               "ernie45-n10"),
    ("meta-llama/llama-4-maverick",             "llama4-maverick-n10"),
    ("qwen/qwen3-235b-a22b",                    "qwen3-235b-n10"),
    ("deepseek/deepseek-r1",                    "deepseek-r1-n10"),
    ("deepseek/deepseek-r1-0528",               "deepseek-r1-0528-n10"),
    ("qwen/qwen3-235b-a22b-thinking-2507",      "qwen3-235b-thinking-n10"),
    ("google/gemini-2.5-pro",                   "gemini25-pro-n10"),
    ("openai/o1",                               "o1-n10"),
    ("openai/o3",                               "o3-n10"),
    ("meta-llama/llama-3.2-3b-instruct",        "llama32-3b-n10"),
]

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
CSV_PATH   = ROOT.parent / "llm_eval_list_v2.csv"
RESULTS    = ROOT / "results" / "syco"
BENCH      = ROOT / "../new-eval/sycophancy_eval_csv_full.json"
SCRIPT     = ROOT / "syco_experiment.py"
BATCH_LOG  = ROOT / "run_syco_batch.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(BATCH_LOG),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── CSV helpers ───────────────────────────────────────────────────────────────
def set_status(openrouter_id: str, status: str):
    rows = list(csv.DictReader(open(CSV_PATH)))
    fieldnames = list(rows[0].keys())
    for row in rows:
        if row.get("OpenRouter_ID", "").strip() == openrouter_id:
            row["Study_2_Syco_Complete"] = status
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    total = len(MODELS)
    log.info(f"Batch starting — {total} models queued")

    for i, (model_id, slug) in enumerate(MODELS, 1):
        output = RESULTS / slug
        model_log = ROOT / f"run_syco_{slug}.log"

        log.info(f"[{i}/{total}] START  {model_id}")
        set_status(model_id, "In Progress")

        cmd = [
            sys.executable, str(SCRIPT),
            "--model", model_id,
            "--runs", "10",
            "--output", str(output),
            "--benchmark", str(BENCH),
        ]

        t0 = time.time()
        with open(model_log, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

        elapsed = time.time() - t0
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)

        if result.returncode == 0:
            log.info(f"[{i}/{total}] DONE   {model_id}  ({mins}m {secs}s)")
            set_status(model_id, "Yes")
        else:
            log.warning(f"[{i}/{total}] FAILED {model_id}  (exit {result.returncode}) — check {model_log.name}")
            set_status(model_id, "Failed")

    log.info("Batch complete.")

if __name__ == "__main__":
    main()
