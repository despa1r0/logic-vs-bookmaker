from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "matches.csv"
OUT = ROOT / "data" / "features.csv"

FORM_WINDOW = 5
FORM_MIN_GAMES = 3

ELO_START = 1500
ELO_NEW_TEAM = 1450
ELO_K = 20
ELO_HOME_ADV = 65
ELO_SEASON_REGRESSION = 0.25

TRAIN_SEASONS = ["2016/17", "2017/18", "2018/19", "2019/20", "2020/21", "2021/22", "2022/23"]
VALID_SEASONS = ["2023/24"]
TEST_SEASONS = ["2024/25", "2025/26"]

RESULT_CODE = {"H": 0, "D": 1, "A": 2}


def to_team_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (home and away)."""
    home = pd.DataFrame({
        "match_id": df["match_id"], "date": df["date"], "team": df["home_team"],
        "goals_for": df["home_goals"], "goals_against": df["away_goals"],
        "sot_for": df["home_shots_on_target"], "sot_against": df["away_shots_on_target"],
        "points": df["result"].map({"H": 3, "D": 1, "A": 0}),
    })
    away = pd.DataFrame({
        "match_id": df["match_id"], "date": df["date"], "team": df["away_team"],
        "goals_for": df["away_goals"], "goals_against": df["home_goals"],
        "sot_for": df["away_shots_on_target"], "sot_against": df["home_shots_on_target"],
        "points": df["result"].map({"A": 3, "D": 1, "H": 0}),
    })
    return pd.concat([home, away]).sort_values(["team", "date", "match_id"])


def add_form(df: pd.DataFrame) -> pd.DataFrame:
    stats = ["goals_for", "goals_against", "sot_for", "sot_against", "points"]
    rows = to_team_rows(df)
    # shift(1): the current match never sees its own result
    form = (
        rows.groupby("team")[stats]
        .transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=FORM_MIN_GAMES).mean())
        .add_prefix("form_")
    )
    rows = pd.concat([rows[["match_id", "team"]], form], axis=1)

    for side in ("home", "away"):
        part = rows.rename(columns={"team": f"{side}_team"})
        part = part.rename(columns={c: f"{side}_{c}" for c in form.columns})
        df = df.merge(part, on=["match_id", f"{side}_team"], how="left")

    for stat in stats:
        df[f"diff_form_{stat}"] = df[f"home_form_{stat}"] - df[f"away_form_{stat}"]
    return df


def goal_multiplier(goal_diff: int) -> float:
    n = abs(goal_diff)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11 + n) / 8


def add_elo(df: pd.DataFrame) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    prev_season_teams: set[str] = set()
    elo_home, elo_away = [], []

    for season, games in df.groupby("season", sort=False):
        teams = set(games["home_team"]) | set(games["away_team"])
        for team in teams:
            if team in prev_season_teams:
                ratings[team] = (1 - ELO_SEASON_REGRESSION) * ratings[team] + ELO_SEASON_REGRESSION * ELO_START
            else:
                ratings[team] = ELO_NEW_TEAM if prev_season_teams else ELO_START

        for row in games.itertuples():
            r_h, r_a = ratings[row.home_team], ratings[row.away_team]
            elo_home.append(r_h)
            elo_away.append(r_a)

            expected_home = 1 / (1 + 10 ** (-(r_h + ELO_HOME_ADV - r_a) / 400))
            actual_home = {"H": 1.0, "D": 0.5, "A": 0.0}[row.result]
            delta = ELO_K * goal_multiplier(row.home_goals - row.away_goals) * (actual_home - expected_home)
            ratings[row.home_team] = r_h + delta
            ratings[row.away_team] = r_a - delta

        prev_season_teams = teams

    df["elo_home"] = elo_home
    df["elo_away"] = elo_away
    df["elo_diff"] = df["elo_home"] + ELO_HOME_ADV - df["elo_away"]
    df["elo_expected_home"] = 1 / (1 + 10 ** (-df["elo_diff"] / 400))
    return df


def add_bookmaker_probs(df: pd.DataFrame) -> pd.DataFrame:
    inv = 1 / df[["odds_home", "odds_draw", "odds_away"]]
    total = inv.sum(axis=1)
    df["bookmaker_margin"] = total - 1
    df["bk_prob_home"] = inv["odds_home"] / total
    df["bk_prob_draw"] = inv["odds_draw"] / total
    df["bk_prob_away"] = inv["odds_away"] / total
    return df


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    df["split"] = np.select(
        [df["season"].isin(TRAIN_SEASONS), df["season"].isin(VALID_SEASONS), df["season"].isin(TEST_SEASONS)],
        ["train", "valid", "test"],
        default="unused",
    )
    df["target"] = df["result"].map(RESULT_CODE)
    return df


def baseline_report(df: pd.DataFrame) -> None:
    probs = df[["bk_prob_home", "bk_prob_draw", "bk_prob_away"]].to_numpy()
    y = df["target"].to_numpy()
    p_true = np.clip(probs[np.arange(len(y)), y], 1e-15, 1)
    df = df.assign(_ll=-np.log(p_true), _hit=(probs.argmax(axis=1) == y), _home=(y == 0))

    print("\nBaselines (log loss lower is better):")
    for split in ("train", "valid", "test"):
        part = df[df["split"] == split]
        print(
            f"  {split:5s} n={len(part):4d} | bookmaker log loss={part['_ll'].mean():.4f} "
            f"acc={part['_hit'].mean():.3f} | always-home acc={part['_home'].mean():.3f} "
            f"| uniform log loss={np.log(3):.4f}"
        )


def main() -> None:
    df = pd.read_csv(IN, parse_dates=["date"]).sort_values(["date", "match_id"]).reset_index(drop=True)
    df = add_form(df)
    df = add_elo(df)
    df = add_bookmaker_probs(df)
    df = add_split(df)

    form_cols = [c for c in df.columns if "form_" in c]
    print(f"Rows: {len(df)}, rows with missing form (early matches): {df[form_cols].isna().any(axis=1).sum()}")
    print("Split sizes:", df["split"].value_counts().to_dict())
    print(f"Mean bookmaker margin: {df['bookmaker_margin'].mean():.3%}")
    baseline_report(df)

    df.to_csv(OUT, index=False)
    print(f"\nSaved {OUT.name}: {df.shape}")


if __name__ == "__main__":
    main()
