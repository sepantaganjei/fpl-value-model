# FPL Value Model

Predict what a Fantasy Premier League player's price *should* be from
their underlying performance, then flag who the market has mispriced.

## User

An active Fantasy Premier League manager deciding which players to
transfer in or out.

## Questions

1. Which players are most underpriced right now (predicted price far
   above actual price)?
2. Which players in my squad are overpriced and worth selling?
3. Which position and price tier offers the best points per million?
4. Does the model's value signal predict future points better than
   FPL's own price does?
5. Who are the cheap or newly promoted differential value picks?

## Population

Every player in the current Fantasy Premier League player pool.

## Prediction moment

Any time during the season; refreshed daily.

## Target

`now_cost` -- the player's current FPL price, stored by the API in
tenths of a million (e.g. `75` = GBP 7.5m).

## Features (first version)

Per-90 rates of: goals, assists, expected goals, expected assists,
expected goal involvements, ICT index, bonus, clean sheets, saves.
Plus season minutes (a role/availability proxy) and position
(GK / DEF / MID / FWD, one-hot encoded).

Player-seasons under 450 minutes are dropped -- their per-90 rates are
too noisy to price against. Live players are scored on their most recent
completed season's per-90 form.

## Model output

For every player:

- Predicted price
- Actual price
- Value signal = predicted - actual (positive = underpriced)
- Position, team, points per million

## Model

Start with ordinary least squares linear regression. Compare against
Ridge, Lasso, and a gradient-boosted tree baseline.

## Data sources

- **Live:** `https://fantasy.premierleague.com/api/bootstrap-static/`
  (free, no key, updates daily)
- **Historical:** `vaastav/Fantasy-Premier-League` GitHub repo,
  `data/<season>/players_raw.csv` (seasons 2022-23 onward, where
  expected-goals columns exist)

## Success criteria

The value signal must beat these baselines at picking players whose
points per million improves over the next five gameweeks:

1. Ranking players by raw FPL form
2. Ranking players by lowest price
3. Ranking players by season points per 90 minutes

## Evaluation

- Mean absolute error of predicted price, in GBP millions
- R-squared
- Backtest: forward points per million of the top-N underpriced picks
  versus each baseline

## Known caveat

FPL price is partly driven by transfer activity (prices rise when many
managers buy a player), not only on-pitch performance. The residual
therefore captures both genuine mispricing and price-change lag. This
is acceptable -- surfacing both is the point -- but it is documented so
results are read correctly.
