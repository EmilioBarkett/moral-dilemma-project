# LLM Moral Dilemma Project

Do large language models make moral judgments that track the moral structure of a situation, and do they stand by those judgments when a user pushes back? This project pairs a measure of models' intrinsic moral sensitivity (Study 1) with **MORAL COMPASS**, a benchmark for moral sycophancy (Study 2), across frontier LLMs.

**Status:** in progress. A paper is in preparation and there is no preprint yet.

Began as a [SPAR](https://sparai.org/) research project, was extended during ARENA 8.0, and was presented at the Cooperative AI Summer School.

## Research question

1. **Moral sensitivity.** When a harmful action is given a strong utilitarian justification, do LLMs endorse it more, and by how much compared with humans?
2. **Moral sycophancy.** Do LLMs abandon a principled moral position under social pressure, and can they tell when that pressure is epistemically irrational?

Both are examined across moral foundations, and with attention to whether extended chain-of-thought reasoning changes the picture.

## Studies

### Study 1: Moral sensitivity (process dissociation)

An adaptation of the [Conway & Gawronski (2013)](https://doi.org/10.1037/a0031021) process dissociation paradigm, organized around [Moral Foundations Theory](https://doi.org/10.1037/a0015141).

- **80 dilemmas** in a 5 × 8 × 2 design: five moral foundations (Care, Fairness, Loyalty, Authority, Purity) × eight domains (economic, historical, law, medical, military, personal, science, transport) × two variants.
- **Congruent** variants have no utilitarian justification, so both moral frameworks oppose the harm. **Incongruent** variants add a strong utilitarian justification that conflicts with the deontological prohibition.
- **Moral sensitivity gap** = harm endorsement in incongruent minus congruent variants. Each model answers three questions per dilemma: harm endorsement (Yes/No), likelihood of acting (1–7), and confidence (1–7).
- Run on 27 frontier models at n = 50 trials each, split into reasoning and non-reasoning models.

### Study 2: MORAL COMPASS (moral sycophancy)

Most sycophancy benchmarks test factual capitulation, such as agreeing with a false statement when pushed. MORAL COMPASS, developed by the project team, tests moral capitulation: does a model abandon a moral position under social pressure?

- **340 items** across Haidt's five moral foundations: a 300-item main battery (5 foundations × 5 domains × 2 utility levels × 2 framings × 3 pressure levels) plus 40 control items that anchor the rating scale.
- **Three escalating pressure levels.** L1 is a neutral ask, L2 a leading presupposition ("... isn't it?"), and L3 a sanity appeal ("I'm not crazy, right?").
- **Plain and delusional framings.** The delusional framing tests whether a model can detect that the pressure is epistemically irrational.
- **Primary metric, ΔL2.** The shift in a model's 1–7 moral endorsement rating from the neutral baseline (L1) to leading pressure (L2). A positive value means the model endorsed the transgression more under pressure.
- **Trials.** Each model is run at n = 10 trials per item.

Together with Study 1, this characterizes both how models represent moral information and whether they defend those representations under pressure.

### Comparison: SycEval replication

`comparison_evals/syceval/` replicates [SycEval](https://arxiv.org/abs/2502.08177) (Fanous et al., 2025), which measures sycophancy on math (AMPS) and medical (MedQuad) questions. It serves as a convergent-validity comparison for Study 2.

## Repository layout

```
moral-dilemma-project/
├── study1_spar/                        # Study 1: moral sensitivity (process dissociation)
│   ├── stimuli/                        # The 80 dilemmas
│   ├── experiment.py                   # Runner
│   ├── analyze.py, plot_*.py           # Analysis and figures
│   ├── resume_run.py, reparse_trials.py# Run and parsing utilities
│   ├── analysis/  plots/               # Summary tables, stats report, figures
│   └── results/                        # Per-model output CSVs
├── study2_moral_compass/               # Study 2: moral and delusional sycophancy
│   ├── benchmark/                      # Benchmark JSON and materials
│   ├── experiment.py, run_batch.py     # Runner and sequential batch runner
│   └── results/                        # Per-model output CSVs
├── comparison_evals/syceval/           # SycEval replication, used as a comparison point
├── data/llm_eval_list_v2.csv           # Registry of 64 models
└── archive/                            # Earlier scripts, results, and report
```

Dilemma IDs follow `{foundation}_{domain}_{variant}`, where variant is `con` or `inc` (for example `authority_economic_con`). Each run writes `_trials.csv` (one row per API call), `_stats.csv` (per dilemma), and `_summary.csv` (per foundation).

## Project team

- **Roshni Lulla**, Institute for Humane Robotics
- **Kristin Witte**, Institute for Human-Centered AI, Helmholtz Munich, and Ludwig-Maximilians-Universität München
- **Saloni Modi**, Independent
- **Emilio Barkett**, Columbia University

## References

- Conway, P., & Gawronski, B. (2013). Deontological and utilitarian inclinations in moral decision making: A process dissociation approach. *Journal of Personality and Social Psychology, 104*(2), 216–235.
- Fanous, A., Goldberg, J., Agarwal, A. A., et al. (2025). SycEval: Evaluating LLM Sycophancy. arXiv:2502.08177.
- Graham, J., Haidt, J., & Nosek, B. A. (2009). Liberals and conservatives rely on different sets of moral foundations. *Journal of Personality and Social Psychology, 96*(5), 1029–1046.
