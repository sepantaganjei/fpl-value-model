# FPL Value Model

Predict what a Fantasy Premier League player's price *should* be from
their underlying performance, then flag the players the market has
mispriced.

Every FPL manager stares at the same question before a transfer: is this
player actually worth their price? This project answers it with a model
instead of a gut feeling. It learns the relationship between on-pitch
output (goals, assists, expected goals, minutes, ICT, bonus, ...) and
FPL price across past seasons, then applies that to the current player
pool. The gap between predicted and actual price is the **value
signal**: a large positive gap means the player looks underpriced.

See [`PROJECT.md`](PROJECT.md) for the full problem definition and
evaluation plan.

## Questions it answers

1. Which players are most underpriced right now?
2. Which players in my squad are overpriced and worth selling?
3. Which position and price tier offers the best points per million?
4. Does the value signal predict future points better than FPL's own
   price?
5. Who are the cheap or newly promoted differential value picks?

## Data

| Source | Use | Access |
| --- | --- | --- |
| [`bootstrap-static`](https://fantasy.premierleague.com/api/bootstrap-static/) | Live current-season players | Free, no key, daily |
| [`vaastav/Fantasy-Premier-League`](https://github.com/vaastav/Fantasy-Premier-League) | Historical seasons (2022-23 onward) | Public CSV mirror |

## Caveat

FPL price is partly driven by transfer activity, not only performance:
prices rise when many managers buy a player. The value signal therefore
mixes genuine mispricing with price-change lag. Surfacing both is the
point, but read the output with that in mind.

## Stack

Python 3.12, pandas, scikit-learn, matplotlib, requests, Streamlit.
Managed with Poetry; linted with ruff; type-checked with mypy; tested
with pytest.

## How it works

1. **Features** — counting stats (goals, assists, xG, xA, xGI, ICT,
   bonus, clean sheets, saves) are converted to per-90 rates so a
   part-season and a full season compare fairly. Player-seasons under
   450 minutes are dropped.
2. **Model** — a scikit-learn pipeline (standardise + one-hot position +
   regression). Linear regression is the default; Ridge and a
   gradient-boosted tree are compared in `cross_validate_by_season`.
3. **Scoring** — the fitted model predicts a *fair price* for every
   current player from their most recent season's per-90 form. Live
   players with no history are skipped. `value_m = pred_m - price_m`.
4. **Backtest** — `walk_forward_backtest` fits on earlier seasons,
   drafts the top 20 by each strategy from the prior season, and scores
   them on the next season's points per million.

## Getting started

```bash
poetry install
make data      # fetch and cache FPL data
make train     # fit the model, write data/predictions.parquet
make check     # lint, type-check, test
```

## Project layout

```
src/fpl_value_model/
  config.py      paths, URLs, feature/target column lists
  data.py        fetch + cache live and historical player data
  features.py    per-90 rates, minutes filter, history join
  model.py       pipeline, CV comparison, ValueModel (fit/predict/save)
  backtest.py    walk-forward strategy comparison
  pipeline.py    end-to-end run -> data/predictions.parquet
notebooks/       exploratory companion to the package
tests/           unit tests (pytest)
app/             Streamlit dashboard (Milestone 3)
.github/workflows/  CI and the daily retrain job
```

## Results so far

Season-held-out CV on 1,608 player-seasons (2022-23 to 2025-26):

| model | MAE (GBP m) | R2 |
| --- | --- | --- |
| gradient boosting | 0.54 | 0.55 |
| ridge / linear | 0.56 | 0.53 |
| predict-the-mean | 0.83 | 0.00 |

Backtest, mean forward points per million of the top-20 picks:

| strategy | mean |
| --- | --- |
| model value signal | 19.1 |
| FPL form | 18.9 |
| season points per 90 | 15.0 |
| cheapest | 13.2 |

The value signal edges out raw FPL form and clearly beats the naive
baselines. Premium players (Haaland, Bruno, Salah) always read as
"overpriced" — a linear model on per-90 counting stats can't price the
captaincy/ownership premium.

## Status

- **Milestone 1** — data layer + prototype notebook: done.
- **Milestone 2** — package refactor (`features` / `model` / `backtest`
  / `pipeline`), pytest suite, backtest: done.
- **Milestone 3** — Streamlit dashboard + deploy + daily GitHub Actions
  retrain: next.
