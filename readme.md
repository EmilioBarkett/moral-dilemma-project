# LLM Moral Dilemma Project

A moral psychology experiment studying how large language models (LLMs) respond to structured moral dilemmas across multiple foundations and domains. The project systematically tests whether LLMs exhibit the same congruent/incongruent moral sensitivity observed in human participants.

---

## Background

This project is grounded in **Moral Foundations Theory** (Haidt) and the **Conway & Gawronski (2013)** dilemma paradigm. Each dilemma presents a scenario in two variants:

- **Congruent** — the harmful action violates a moral rule *and* the justification is weak (low stakes). Human participants and well-calibrated LLMs should say **No**.
- **Incongruent** — the harmful action violates a moral rule *but* the justification is strong (high stakes). This is where moral conflict lives. Participants may say **Yes** more often.

The gap between congruent and incongruent endorsement rates is the primary measure of moral sensitivity. A model with no moral reasoning would show no gap. A model reasoning like humans should show a substantial one.

### Moral Foundations Covered

| Foundation | Core concern |
|---|---|
| **Authority** | Obedience vs. disobeying orders |
| **Care** | Harm to individuals vs. greater good |
| **Fairness** | Justice, rights, and equal treatment |
| **Loyalty** | In-group obligations vs. broader ethics |
| **Purity** | Sanctity, dignity, and taboo violations |

Each foundation is tested across **8 domains**: economic, historical, law, medical, military, personal, science, transport.

---

## Project Structure

```
moral-dilemma-project/
│
├── moral_dilemmas_survey_revised.csv   # Main dilemma dataset (80 dilemmas)
├── llm_eval_list_v2.csv                # Registry of 61 LLMs to evaluate
├── .env                                # API key (not committed to git)
│
├── experiment-v3/                      # Current experiment
│   ├── experiment-v3.py                # Main script
│   └── results/                        # Output CSVs land here
│
├── old-eval-v1/                        # Archived results (v1 script)
├── old-eval-v2/                        # Archived results (v2 script)
├── old-eval-v3/                        # Archived results (v3 script)
├── experiment-v2.py                    # Previous experiment script (OpenRouter)
├── batch_gpt_testing_basemodels.py     # Early script (OpenAI API only)
└── dilemmas.csv                        # Old dilemma format (10 dilemmas)
```

---

## Datasets

### `moral_dilemmas_survey_revised.csv`
The primary stimulus set. **80 dilemmas** structured as:

```
5 foundations × 8 domains × 2 variants = 80 rows
```

| Column | Description |
|---|---|
| `variable_name` | ID encoding foundation, domain, variant — e.g. `authority_economic_con` |
| `Text Input` | Full scenario text with embedded question |

The `variable_name` convention is `{foundation}_{domain}_{variant}` where variant is `con` (congruent) or `inc` (incongruent).

### `llm_eval_list_v2.csv`
A registry of **61 LLMs** selected for evaluation, spanning multiple organizations, geographies, sizes, and reasoning architectures.

| Column | Description |
|---|---|
| `Model_Name` | Human-readable name |
| `Organization` | Creator (OpenAI, Anthropic, DeepSeek, etc.) |
| `Country` | Country of origin |
| `Tier` | Frontier / Open-Frontier / Open-Large / Open-Small / Legacy |
| `Approx_Size` | Parameter count estimate |
| `Release_Date` | YYYY-MM |
| `Access_Type` | API/Closed or Open Weights |
| `Reasoning_Model` | Yes / No — whether model uses extended chain-of-thought |
| `OpenRouter_ID` | Model ID for use with the experiment script |
| `Notes` | Additional context |

Models marked `N/A` in `OpenRouter_ID` are not currently available via OpenRouter.

---

## Running the Experiment

### Setup

**1. Install dependencies**
```bash
pip install requests python-dotenv
```

**2. Set your API key**

Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_key_here
```
Get a key at [openrouter.ai](https://openrouter.ai).

**3. Navigate to the experiment folder**
```bash
cd experiment-v3
```

### Basic Usage

```bash
# Single model, 5 runs (quick test)
python experiment-v3.py --model "openai/gpt-4o-mini" --runs 5 --output results/test

# Single model, full research run
python experiment-v3.py --model "openai/gpt-4o" --runs 30 --output results/gpt4o-n30

# Multiple models in sequence
python experiment-v3.py --models "openai/gpt-4o,anthropic/claude-sonnet-4" --runs 30 --output results/multi-n30

# Yes/No only (no likelihood or confidence ratings)
python experiment-v3.py --model "openai/gpt-4o-mini" --runs 10 --simple --output results/simple-test

# List all available models on OpenRouter
python experiment-v3.py --list-models
```

### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--model`, `-m` | `openai/gpt-4o` | Single OpenRouter model ID |
| `--models` | — | Comma-separated list of model IDs to run in sequence |
| `--dilemmas`, `-d` | `../moral_dilemmas_survey_revised.csv` | Path to dilemma CSV |
| `--runs`, `-r` | `5` | Runs per dilemma — use 5 to test, 30–50 for real data |
| `--temperature`, `-t` | `0.9` | Sampling temperature |
| `--output`, `-o` | `results/results` | Output file prefix |
| `--seed` | `42` | Random seed for presentation order |
| `--api-key`, `-k` | — | OpenRouter key (or use `.env`) |
| `--delay` | `0.25` | Seconds between API calls |
| `--simple`, `-s` | — | Ask Yes/No only (no likelihood or confidence) |
| `--list-models` | — | Print available models and exit |
| `--verbose`, `-v` | — | Enable debug logging |

### How Many Runs?

| Runs | Total calls | Time (est.) | Use case |
|---|---|---|---|
| 1 | 80 | ~2 min | Smoke test |
| 10 | 800 | ~18 min | Pilot / sanity check |
| 30 | 2,400 | ~55 min | Good research quality |
| 50 | 4,000 | ~90 min | Publication quality |

---

## Output Files

Every run produces three CSV files at the specified `--output` prefix.

### `_trials.csv`
One row per API call. The full raw record.

| Column | Description |
|---|---|
| `run` | Run number (1 to n_runs) |
| `position` | Presentation order within the run |
| `model` | OpenRouter model ID |
| `dilemma_id` | e.g. `authority_economic_con` |
| `foundation` | authority / care / fairness / loyalty / purity |
| `domain` | economic / historical / law / medical / military / personal / science / transport |
| `variant` | congruent / incongruent |
| `raw_response` | Exact text the model returned |
| `endorsement` | Parsed yes / no |
| `likelihood` | Parsed 1–7 score |
| `confidence` | Parsed 1–7 score |
| `fully_valid` | True if all three fields parsed successfully |

### `_stats.csv`
One row per dilemma (80 rows for a single model). All runs aggregated.

Key columns: `pct_harm_endorsement`, `mean_likelihood`, `sd_likelihood`, `mean_confidence`, `sd_confidence`, `n_invalid`.

### `_summary.csv`
One row per moral foundation + one overall row (6 rows for a single model). The highest-level view — use this for charts and cross-model comparisons.

Key columns: `mean_harm_endorsement_congruent`, `se_harm_endorsement_congruent`, `mean_harm_endorsement_incongruent`, `se_harm_endorsement_incongruent`.

---

## Measures

The experiment asks three questions per dilemma:

1. **Harm Endorsement** — *Is the described action appropriate? (Yes or No)* — the primary moral judgment measure
2. **Likelihood** — *How likely are you to break the moral rule? (1–7)* — reflects perceived rule violation
3. **Confidence** — *How confident are you in your answer? (1–7)* — measures decisiveness

The key dependent variable is **% Harm Endorsement** (`pct_harm_endorsement`) — the percentage of runs in which the model said Yes to a given dilemma.

---

## Results So Far

| Model | Runs | Congruent | Incongruent | Gap |
|---|---|---|---|---|
| `openai/gpt-4o-mini` | 10 | 20.0% | 75.25% | 55.25 pts |

### Foundation breakdown (GPT-4o-mini, n=10)

| Foundation | Congruent | Incongruent |
|---|---|---|
| Authority | 37.5% | 100.0% |
| Care | 21.25% | 25.0% |
| Fairness | 0.0% | 80.0% |
| Loyalty | 0.0% | 75.0% |
| Purity | 41.25% | 96.25% |

**Key observations:**
- Fairness and Loyalty show the cleanest congruent/incongruent separation
- Care is nearly flat — the model treats harm-to-persons dilemmas as mostly impermissible regardless of stakes
- Authority congruent is unusually high (37.5%), suggesting some willingness to disobey authority even for weak reasons
- Confidence is consistently high (~6/7) across all conditions

---

## Theoretical Notes

- The **congruent/incongruent gap** is the primary signal. A larger gap = more sensitivity to moral stakes.
- **Reasoning models** (o1, o3, DeepSeek-R1, etc.) may show different patterns due to extended chain-of-thought — a key comparison in this study.
- **Care foundation flatness** in GPT-4o-mini may reflect RLHF training that strongly penalises harm endorsement, potentially overriding utilitarian reasoning.
- Results from a single model at n=10 should be treated as preliminary. Target n=30–50 for stable estimates.

---

## Next Steps

- [ ] Run additional models using `--models` with OpenRouter IDs from `llm_eval_list_v2.csv`
- [ ] Increase to n=30–50 runs for publication-quality data
- [ ] Build visualisation script for cross-model comparison bar charts
- [ ] Compare reasoning vs. non-reasoning models on the same dilemmas
- [ ] Analyse cross-cultural patterns (US vs. Chinese vs. European labs)
