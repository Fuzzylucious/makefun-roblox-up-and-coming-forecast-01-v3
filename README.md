# Roblox Up-and-Coming top-10 forecaster v3

Monte Carlo rank-diffusion forecaster for the make.fun bounty
`roblox-up-and-coming-forecast-01` ("Predict Roblox's Up-and-Coming
top 10", $500, top 20 paid). It reads the chart's history from
`/input/history.ndjson` + `/input/task.json` and writes
`/output/forecast.json`: per-place probabilities for the top 10 places
plus a top-10 guess in order, scored by log score over the 72 hours
after entries close.

## Method

Monte Carlo diffusion in latent-appeal space (`appeal = -log rank`),
1500 seeded paths, 72 hourly steps:

- **Drift** — OLS regression on 3,650 consecutive-pair transitions
  from the archive (R² = 0.134):
  `E[da/h] = -0.049 + 0.546·(appeal slope, trailing 96 h) + 0.625·(log-playing
  slope, trailing 30 h) + 0.0116·log(rank) + 0.0007·log(chart-age h)
  + 0.0044·(log(playing) − median log(playing) of the last chart)`,
  scaled by 1.2. The playing-**level** term is v3's new feature: a
  game with far more players than its rank peers is breaking out even
  before its rank slope moves.
- **Noise** — Student-t steps (df 12), flat scale 0.10 per √hour.
- **Entrants** — phantom games drawn from the empirical entry-rank
  distribution (~0.5/hour), climbing with a decaying drift; they carry
  the probability mass of games that do not exist yet, which the
  scorer leaves unnamed and spreads over the chart's unnamed games.
- **No age-out** — there is no 28-day cliff (games 42–44 days old
  still chart); old games fade through the drift's rank term.
- **Persistence blend (10%)** — the Monte Carlo forecast is averaged
  with the starter persistence forecast (0.5 on the game at each
  place, 0.15 on its neighbours).

## Validation (holdout windows)

Fit on snapshots 1..cut, scored on the reads in the 72 h after the
cut, cuts 20/24/28/30/34/38/42 (full 46-read archive):

| version | mean | median | recent-4 | practice window (cut34) |
|---------|------|--------|----------|-------------------------|
| v2 (live: −31.4, rank 13) | −34.80 | −32.56 | −32.01 | −31.38 |
| **v3** | **−34.42** | **−31.87** | **−31.37** | **−30.78** |

Live practice board at submission time: rank 1 −29.0, rank 10 −30.9,
rank 20 −31.7. v3's practice-window backtest (−30.78) sits just
inside the top 10.

## What was tried and dropped

24 h appeal slope (noise on sparse reads), playing-acceleration
gp30/gp96, tenure/creation-age hazard, rank-band sigma, drift decay,
temperature flattening, ensembling with a slope-drift MC, and a
fuller regression (R² 0.156 in-sample) — all worse on holdouts.
In-sample R² does not transfer; only the playing-level feature and
the persistence blend survived.

## Run

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, no network, deterministic (fixed seed):
two runs produce byte-identical output.
