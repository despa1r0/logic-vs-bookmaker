"""Переименовывает колонки epl_raw.csv в понятные названия -> epl_clean.csv"""
import pandas as pd

RENAME = {
    "Season": "season", "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    "HS": "home_shots", "AS": "away_shots", "HST": "home_shots_on_target", "AST": "away_shots_on_target",
    "HC": "home_corners", "AC": "away_corners", "HY": "home_yellow_cards", "AY": "away_yellow_cards",
    "HR": "home_red_cards", "AR": "away_red_cards",
    "B365H": "bet365_odds_home_win", "B365D": "bet365_odds_draw", "B365A": "bet365_odds_away_win",
}

df = pd.read_csv("epl_raw.csv").rename(columns=RENAME)
df["season"] = df["season"].astype(str).map(lambda s: f"20{s[:2]}/{s[2:]}")
df.to_csv("epl_clean.csv", index=False)
print(df.shape)
