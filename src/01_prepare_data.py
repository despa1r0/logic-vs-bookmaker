from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "epl_raw.csv"
OUT = ROOT / "data" / "matches.csv"

RENAME = {
    "Season": "season", "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_shots_on_target", "AST": "away_shots_on_target",
    "HC": "home_corners", "AC": "away_corners",
    "HY": "home_yellow_cards", "AY": "away_yellow_cards",
    "HR": "home_red_cards", "AR": "away_red_cards",
    "B365H": "odds_home", "B365D": "odds_draw", "B365A": "odds_away",
}


def season_label(code) -> str:
    code = str(code).zfill(4)
    return f"20{code[:2]}/{code[2:]}"


def main() -> None:
    df = pd.read_csv(RAW).rename(columns=RENAME)[list(RENAME.values())]
    df["season"] = df["season"].map(season_label)
    df["date"] = pd.to_datetime(df["date"])

    print(f"Rows: {len(df)}")
    missing = df.isna().sum()
    print("Missing values:", missing[missing > 0].to_dict() or "none")

    dupes = df.duplicated(subset=["date", "home_team", "away_team"]).sum()
    print(f"Duplicate matches: {dupes}")

    bad_result = ~df["result"].isin(["H", "D", "A"])
    odds_cols = ["odds_home", "odds_draw", "odds_away"]
    bad_odds = (df[odds_cols] <= 1).any(axis=1)
    df = df[~bad_result & ~bad_odds].dropna().drop_duplicates(subset=["date", "home_team", "away_team"])

    df = df.sort_values(["date", "home_team"]).reset_index(drop=True)
    df.insert(0, "match_id", range(len(df)))

    print("Matches per season:", df.groupby("season").size().to_dict())
    print("Result share:", df["result"].value_counts(normalize=True).round(3).to_dict())

    df.to_csv(OUT, index=False)
    print(f"Saved {OUT.name}: {df.shape}")


if __name__ == "__main__":
    main()
