"""
smoke_test_gpt6.py
------------------
Sends 5 benchmark items to GPT-6 Astra (openai/gpt-6-astra) via OpenRouter
with max_tokens=10, exactly as syco_experiment.py would, and prints the raw
response + parsed result for each call.

Goal: confirm the model returns a plain 1-7 digit rather than verbose text
or empty content (which would indicate reasoning token overflow).

Usage:
  cd experiment-v3
  python smoke_test_gpt6.py
"""

import json
import os
import re
import sys

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

MODEL      = "openai/gpt-6-astra"
MAX_TOKENS = 200
BENCHMARK  = os.path.join(os.path.dirname(__file__), "../new-eval/sycophancy_eval_csv_full.json")
N_ITEMS    = 5
API_URL    = "https://openrouter.ai/api/v1/chat/completions"


def parse_rating(raw):
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\b([1-7])\b", cleaned)
    if m:
        return int(m.group(1))
    m = re.search(r"[1-7]", cleaned)
    if m:
        return int(m.group(0))
    return None


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("No OPENROUTER_API_KEY found in environment.")

    if not os.path.isfile(BENCHMARK):
        sys.exit(f"Benchmark not found: {BENCHMARK}")

    with open(BENCHMARK, encoding="utf-8") as fh:
        items = json.load(fh)

    sample = items[:N_ITEMS]

    print(f"\nSmoke test — model: {MODEL} | max_tokens: {MAX_TOKENS} | n: {N_ITEMS}")
    print("=" * 70)

    passed = 0
    for i, item in enumerate(sample, 1):
        prompt = item["input"]
        payload = {
            "model":       MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 1.0,
            "max_tokens":  MAX_TOKENS,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }

        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data    = r.json()
            msg     = data["choices"][0]["message"]
            content = msg.get("content")
            if content is None:
                raw = f"[NULL — refusal: {msg.get('refusal', '')}]"
            else:
                raw = content
        except Exception as e:
            raw = f"[ERROR: {e}]"

        parsed = parse_rating(raw) if not raw.startswith("[") else None
        status = "PASS" if parsed is not None else "FAIL"
        if parsed is not None:
            passed += 1

        print(f"\n[{i}] item_id: {item['id']}  →  {status}")
        print(f"    raw    : {repr(raw[:120])}")
        print(f"    parsed : {parsed}")

    print("\n" + "=" * 70)
    print(f"Result: {passed}/{N_ITEMS} parsed successfully.")
    if passed == N_ITEMS:
        print("GPT-6 Astra is safe to run with max_tokens=10.")
    elif passed == 0:
        print("All failed — likely reasoning token overflow or verbose responses.")
        print("Consider passing --max-tokens 200 and re-testing.")
    else:
        print(f"{N_ITEMS - passed} failed — check raw outputs above for pattern.")
    print()


if __name__ == "__main__":
    main()
