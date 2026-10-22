# FPL Predictor

A machine learning and optimization pipeline to predict Fantasy Premier League (FPL) player points and select the highest-scoring squad within the £100m budget.

## Project Overview

Predicting FPL points and picking a team are treated as two connected problems:
1. **Point Prediction (Machine Learning):** Using historical gameweek data and rolling player features to forecast next-gameweek points with `LightGBM`, benchmarked against rolling-average baselines.
2. **Squad Selection (Linear Programming):** Feeding the predicted points into `PuLP` to pick an optimal 15-player squad under official FPL budget, position, and club constraints *(in progress)*.

## Tech Stack

* **Python** (`pandas`, `numpy`, `requests`) for data collection and feature engineering
* **LightGBM** (`lightgbm`) for gradient-boosted point predictions
* **PuLP** (`pulp`) for integer linear programming squad optimization
* **Google Colab / Jupyter Notebooks** for interactive experimentation

## Data Sources

* **Live FPL API:** [`bootstrap-static`](https://fantasy.premierleague.com/api/bootstrap-static/) for current player prices, positions, and clubs.
* **Historical Gameweek Data:** [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) repository (2023–24 season, ~25,000 player-gameweek rows).

*(Note: Raw data files are downloaded directly via URL in the notebooks rather than committed to the repository.)*

## Results & Benchmarks (2023–24 Season)

### 1. Full-Season Baseline (Gameweeks 1–38)
Before training a machine learning model, simple rolling averages (shifted by 1 gameweek using `.shift(1)` to prevent data leakage) were compared against a naive season-average guess:

| Prediction Method | MAE | RMSE | Spearman Rank Corr |
| :--- | :---: | :---: | :---: |
| Always predict the season average | 1.40 | — | — |
| Last 3 games average (`pred_last3`) | 1.01 | 2.14 | — |
| Last 5 games average (`pred_last5`) | **0.99** | **2.08** | **0.69** |

### 2. LightGBM vs. Baseline (Out-of-Sample Test: Gameweeks 26–38)
The `LightGBM` regressor was trained on **Gameweeks 1–25** using 29 pre-match and rolling features (3- and 5-game rolling averages for points, minutes, goals, assists, bonus, BPS, clean sheets, ICT index, xG, xA, plus previous match minutes, position, price, home/away status, and opponent). It was evaluated on unseen data from **Gameweeks 26–38**:

#### Starters Only (`minutes_prev >= 60`)
Evaluating on likely starters removes bench players who score 0 every week and tests how well the model ranks realistic FPL options:

| Model | MAE (↓ better) | RMSE (↓ better) | Rank Corr (↑ better) |
| :--- | :---: | :---: | :---: |
| 5-Game Rolling Baseline (`pred_last5`) | 2.404 | 3.330 | 0.245 |
| **LightGBM (`pred_lgbm`)** | **2.258** | **3.187** | **0.275** |

#### All Players (Including Bench / Non-Starters)

| Model | MAE (↓ better) | RMSE (↓ better) | Rank Corr (↑ better) |
| :--- | :---: | :---: | :---: |
| 5-Game Rolling Baseline (`pred_last5`) | 0.908 | 1.973 | **0.707** |
| **LightGBM (`pred_lgbm`)** | **0.882** | **1.866** | 0.695 |

### Top Features (by Gain)
The most influential signals relied on by the LightGBM model were:
1. `minutes_prev` (minutes played in the previous gameweek)
2. `value` (player price)
3. `opponent_team` (upcoming fixture opponent)
4. `total_points_last3` & `minutes_last3` (short-term form and pitch time)
5. `ict_index_last5` & `ict_index_last3` (underlying attacking involvement)

## Repository Structure

* `notebooks/01_load_data.ipynb` — Fetching live FPL API tables and exploring historical gameweek data.
* `notebooks/02_baseline.ipynb` — Rolling-average baselines, LightGBM feature engineering, out-of-sample evaluation, and feature importance.
* `notebooks/03_baseline.ipynb` — Feature engineering (rolling stats, pre-match context), training the LightGBM model on GW 1–25, out-of-sample       evaluation on GW 26–38 against baseline predictions, and feature importance analysis.
* `LICENSE` — MIT License.

## Next Steps

* [ ] **Step 5:** Build the `PuLP` integer linear programming optimizer to select a valid 15-man squad and starting XI from the model's predictions.
* [ ] Add multi-season training data and opponent strength ratings.
