# FPL Midfielder Return Predictor

## User

An active Fantasy Premier League manager deciding which midfielder
to transfer into their squad.

## Question

Which midfielders are most likely to record at least one goal or
assist in the next gameweek?

## Population

Players classified as midfielders in Fantasy Premier League.

## Prediction moment

Immediately before the gameweek deadline.

## Target

return_next_gw = 1 if goals_scored + assists > 0 in the next
gameweek, otherwise 0.

## Model output

For every midfielder:

- Probability of an attacking return
- Predicted probability of playing at least 60 minutes
- Price
- Next opponent
- Home or away
- Prediction confidence

## First version

The first version will predict attacking returns only. Expected FPL
points, forwards and personalised transfer recommendations will be
added later.

## Success criteria

The model must beat:

1. Ranking players by returns in their previous five matches
2. Ranking players by season returns per 90 minutes
3. Ranking players by FPL form
4. A simple logistic-regression baseline

## Evaluation

- Log loss
- Brier score
- ROC AUC
- Precision among the top 10 recommendations
- Calibration