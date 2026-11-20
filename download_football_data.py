"""Скачивает сезоны Premier League с football-data.co.uk и склеивает в один CSV.

Запуск:  python download_football_data.py
Результат: epl_raw.csv (все сезоны, нужные колонки, отсортировано по дате)
"""
import io
import urllib.request

import pandas as pd

LEAGUE = "E0"  # E0 = Premier League; SP1 = La Liga, D1 = Bundesliga, I1 = Serie A
SEASONS = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

KEEP = [
    "Season", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",          # голы и результат (H/D/A)
    "HS", "AS", "HST", "AST",       # удары и удары в створ
    "HC", "AC", "HY", "AY", "HR", "AR",  # угловые, карточки
    "B365H", "B365D", "B365A",      # коэффициенты Bet365
]


def load_season(season: str) -> pd.DataFrame:
    url = URL.format(season=season, league=LEAGUE)
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read()
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    df["Season"] = season
    return df


def main() -> None:
    frames = []
    for season in SEASONS:
        try:
            frames.append(load_season(season))
            print(f"OK   {season}: {len(frames[-1])} матчей")
        except Exception as exc:  # сезон пропускаем, остальные продолжаем
            print(f"FAIL {season}: {exc}")

    data = pd.concat(frames, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date")
    data = data[[c for c in KEEP if c in data.columns]]
    data.to_csv("epl_raw.csv", index=False)
    print(f"Сохранено epl_raw.csv: {len(data)} строк, {data.shape[1]} колонок")


if __name__ == "__main__":
    main()
