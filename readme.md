# LLM Moral Dilemma Project

Do large language models make moral judgments that track the moral structure of a situation, or ones that depend on how it is framed? This project studies LLM moral reasoning with paradigms from moral psychology, across dozens of frontier models.

**Status:** in progress. A paper is in preparation and there is no preprint yet.

Part of a [SPAR](https://sparai.org/) research project.

## Research question

When a harmful action is given a strong utilitarian justification, do LLMs endorse it more, and by how much compared with humans? Does this change with the moral foundation involved, with extended chain-of-thought reasoning, or with social pressure from the user?

## Studies

### Study 1: Moral sensitivity (process dissociation)

An adaptation of the [Conway & Gawronski (2013)](https://doi.org/10.1037/a0031021) process dissociation paradigm, organized around [Moral Foundations Theory](https://doi.org/10.1037/a0015141).

- **80 dilemmas** in a 5 × 8 × 2 design: five moral foundations (Care, Fairness, Loyalty, Authority, Purity) × eight domains (economic, historical, law, medical, military, personal, science, transport) × two variants.
- **Congruent** variants have no utilitarian justification, so both moral frameworks oppose the harm. **Incongruent** variants add a strong utilitarian justification that conflicts with the deontological prohibition.
- **Moral sensitivity gap** = harm endorsement in incongruent minus congruent variants. Each model answers three questions per dilemma: harm endorsement (Yes/No), likelihood of acting (1–7), and confidence (1–7).
- Run on 27 models at n = 50 trials each, split into reasoning and non-reasoning models.

### Study 2: Moral and delusional sycophancy

A 340-item benchmark (Lulla & Witte, 2026) testing whether a model's moral ratings shift when the user applies conversational pressure (a neutral question, a leading presupposition, then an appeal such as "I'm not crazy, right?"). Run on 23 models at n = 10 trials each.

## Repository layout

```
moral-dilemma-project/
├── moral_dilemmas_survey_revised.csv   # Study 1 stimuli: 80 dilemmas
├── llm_eval_list_v2.csv                # Registry of 64 models and per-study progress
├── experiment-v3/
│   ├── experiment-v3.py                # Study 1 runner
│   ├── syco_experiment.py              # Study 2 runner
│   ├── run_syco_batch.py               # Sequential batch runner for Study 2
│   ├── analyze_results.py, plot_*.py   # Analysis and figures
│   ├── resume_run.py, reparse_trials.py# Run and parsing utilities
│   ├── analysis/                       # Summary tables and stats report
│   ├── plots/                          # Figures
│   └── results/                        # Per-model output CSVs
│       ├── reasoning/  non-reasoning/  # Study 1, by model type
│       ├── syco/                       # Study 2
│       └── pilot-1/  abandoned/        # Early and discarded runs
├── new-eval/                           # Study 2 benchmark and materials
├── report/                             # SPAR midterm report
└── old-evals/                          # Earlier scripts and archived results
```

Dilemma IDs follow `{foundation}_{domain}_{variant}`, where variant is `con` or `inc` (for example `authority_economic_con`). Each run writes `_trials.csv` (one row per API call), `_stats.csv` (per dilemma), and `_summary.csv` (per foundation).

## Project team

- **Roshni Lulla**, Institute for Humane Robotics
- **Kristin Witte**, Institute for Human-Centered AI, Helmholtz Munich, and Ludwig-Maximilians-Universität München
- **Saloni Modi**, Independent
- **Emilio Barkett**, Columbia University

## References

- Conway, P., & Gawronski, B. (2013). Deontological and utilitarian inclinations in moral decision making: A process dissociation approach. *Journal of Personality and Social Psychology, 104*(2), 216–235.
- Graham, J., Haidt, J., & Nosek, B. A. (2009). Liberals and conservatives rely on different sets of moral foundations. *Journal of Personality and Social Psychology, 96*(5), 1029–1046.
