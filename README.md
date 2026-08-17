# Premier League Match Prediction Model

A machine learning model that predicts Premier League match outcomes for the 2026/27 season. Built as a self-directed summer project while studying AI at the University of Bergen (UIB).

## What it does

- Predicts the result (home win / draw / away win) for all 380 matches in the 2026/27 Premier League season
- Outputs win probabilities for each match (P(H), P(D), P(A))
- Simulates the full season 1000 times using Monte Carlo simulation to estimate each team's chances of winning the title, finishing top 4, or getting relegated

## How it works

The model uses XGBoost, which builds decision trees sequentially where each tree learns from the mistakes of the previous one. I tested logistic regression and random forest first, but XGBoost gave consistently better accuracy so I went with that.

### Features

| Feature | Description |
|---|---|
| ELO rating | A rating system where teams gain or lose points based on match results. Initialized using total historical points so that clubs like Arsenal and Man Utd start higher than newly promoted sides |
| Recent form | Average points per game over the last 5 matches |
| Goals scored/conceded | Average over the last 5 matches |
| FDR (Fixture Difficulty Rating) | Official Premier League difficulty rating per fixture (1=easy, 5=hard) |
| H2H (Head-to-Head) | Historical win/draw/loss ratio between the two teams |

### Training

- Data: 23 seasons of Premier League data (2003/04 to 2025/26), sourced from [football-data.co.uk](https://www.football-data.co.uk)
- Train/test split: trained on all seasons before 2026, tested on the 2025/26 season
- Accuracy on test set: **37.1%**

### A note on accuracy

37% might look low, but predicting football is genuinely hard. Even professional models rarely get above 55-60% because of how random the sport is. A model that always predicts the home team would get around 45%, but that is not useful. This model predicts all three outcomes and gives probabilities for each, which is more meaningful than just picking a winner.

### Monte Carlo simulation

Rather than always picking the most likely result, the model runs the full season 1000 times. In each run, match results are sampled randomly based on the model's probabilities. This gives a more realistic picture of how the season could play out, for example "Arsenal wins the title in 44% of simulations", rather than a single fixed prediction.

## Limitations

The model is based on historical match data only and does not account for:
- Manager changes between seasons
- Transfer activity and squad changes
- Injuries or suspensions

This means teams going through big changes may be over or underestimated. Adding pre-season odds or transfer data as features would likely improve this.

## Files

| File | Description |
|---|---|
| `model.py` | Main script that loads data, trains the model, and generates predictions |
| `fdr_2027.csv` | Fixture difficulty ratings for all 380 matches in 2026/27 |
| `predictions_2027.csv` | Predicted outcomes and probabilities for all 380 matches |
| `simulated_table_2027.csv` | Simulated league table with title, top 4 and relegation probabilities |

## Tools and libraries

- Python
- pandas, numpy
- XGBoost
- scikit-learn

## About

Built during summer 2026 as a self-directed learning project alongside the Kaggle ML micro-courses and Anthropic's courses on building with AI. Claude was used as a learning aid throughout the process.
