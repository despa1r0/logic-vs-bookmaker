# Logic vs Bookmaker

**RU:** Можно ли предсказать исход футбольного матча (Premier League) лучше, чем букмекер?
**EN:** Can we predict the outcome of a football match (Premier League) better than a bookmaker?

[Русский](#русский) · [English](#english)

---

## Русский

### Структура проекта

```
logic vs bookmaker/
├── data/
│   ├── epl_raw.csv               исходные данные (football-data.co.uk)
│   ├── matches.csv               этап 1: очищенные матчи
│   ├── features.csv              этап 2: признаки + вероятности букмекера + разделение
│   ├── team_season_clusters.csv  этап 3: кластер каждой команды в каждом сезоне
│   └── model_dataset.csv         итоговая таблица для моделей (этап 2 + этап 3)
├── figures/                      графики кластеризации
├── src/
│   ├── 01_prepare_data.py
│   ├── 02_features.py
│   └── 03_clustering.py
└── requirements.txt
```

### Запуск

```
pip install -r requirements.txt
python src/01_prepare_data.py
python src/02_features.py
python src/03_clustering.py
```

Скрипты запускаются по порядку. Этапы 2 и 3 зависят только от `matches.csv`, поэтому их можно делать параллельно.

### Данные

- 10 сезонов Premier League (2016/17 – 2025/26), 3800 матчей, 20 команд в сезоне.
- Для каждого матча: команды, голы, итог, удары, удары в створ, угловые, карточки и коэффициенты.
- **Коэффициенты от одного букмекера: Bet365.** Это не среднее и не медиана по рынку.
- Итог матча (`result`): `H` победа хозяев, `D` ничья, `A` победа гостей. Это то, что предсказываем.

Распределение исходов: хозяева 44.6%, гости 32.1%, ничьи 23.2%.

### Этап 1. Подготовка данных (`01_prepare_data.py`)

1. Переименовывает колонки в понятные названия (`FTHG` → `home_goals` и т.д.).
2. Проверяет пропуски, дубликаты, корректность итога и коэффициентов (коэффициент должен быть > 1).
3. Сортирует по дате и присваивает каждому матчу номер `match_id`.

Результат проверки: пропусков 0, дубликатов 0, в каждом сезоне 380 матчей.

### Этап 2. Признаки (`02_features.py`)

**Главное правило: для матча используем только то, что было известно ДО его начала.**
Если в признак попадёт результат самого матча (или будущих матчей), модель покажет отличную точность, которая в реальности недостижима. Это называется утечкой данных (data leakage).

**2.1. Форма команды** — среднее за последние 5 матчей команды до текущего:

| Признак | Что значит |
|---|---|
| `form_goals_for` | забитые голы |
| `form_goals_against` | пропущенные голы |
| `form_sot_for` / `form_sot_against` | удары в створ (свои / соперника) |
| `form_points` | очки (победа 3, ничья 1, поражение 0) |

Каждый признак есть у хозяев (`home_form_...`), у гостей (`away_form_...`) и как разница (`diff_form_...`).
Нужно минимум 3 прошлых матча, иначе значение пустое. Это 71 матч в самом начале данных, перед обучением модели их нужно удалить.

**2.2. Рейтинг Elo** — сила команды одним числом. После каждого матча рейтинг меняется:

```
ожидание хозяев  E = 1 / (1 + 10^(-(R_хозяев + 65 - R_гостей) / 400))
результат        S = 1 (победа), 0.5 (ничья), 0 (поражение)
изменение        Δ = 20 × G × (S − E)
R_хозяев += Δ,   R_гостей −= Δ
```

- 65 — бонус домашнего поля, 20 — скорость изменения (K).
- G — множитель за разницу голов: 1 (разница 0–1), 1.5 (разница 2), (11 + N)/8 (разница N ≥ 3).
- Старт 1500. Перед новым сезоном рейтинг сдвигается на 25% к 1500 (составы меняются).
- Команда, вышедшая из низшей лиги, начинает с 1450.
- В таблицу записывается рейтинг **до** матча, обновление происходит после.

Колонки: `elo_home`, `elo_away`, `elo_diff` (разница с учётом домашнего поля), `elo_expected_home`.
Проверка: `elo_expected_home` коррелирует с вероятностью букмекера на 0.93, значит рейтинг отражает реальную силу команд.

**2.3. Вероятности букмекера** — эталон, с которым сравниваем свои модели:

```
q = 1 / коэффициент                  для каждого исхода
маржа = q_H + q_D + q_A − 1          (в среднем 4.7%)
вероятность = q / (q_H + q_D + q_A)  (сумма = 1)
```

Колонки: `bk_prob_home`, `bk_prob_draw`, `bk_prob_away`, `bookmaker_margin`.

**2.4. Разделение по времени** (колонка `split`):

| Часть | Сезоны | Матчей | Зачем |
|---|---|---|---|
| `train` | 2016/17 – 2022/23 | 2660 | обучение моделей |
| `valid` | 2023/24 | 380 | подбор параметров |
| `test` | 2024/25 – 2025/26 | 760 | финальная проверка, один раз в конце |

Случайное разделение здесь нельзя: модель «видела бы будущее».
`target` — итог числом: 0 = H, 1 = D, 2 = A.

**Базовые линии** (log loss: чем меньше, тем лучше; accuracy: доля угаданных исходов):

| | Букмекер log loss | Букмекер accuracy | «Всегда хозяева» accuracy | Случайно 1/3, log loss |
|---|---|---|---|---|
| train | 0.947 | 56.2% | 45.3% | 1.099 |
| valid | 0.909 | 59.7% | 46.1% | 1.099 |
| test | 0.995 | 51.4% | 41.7% | 1.099 |

Цель моделей — подойти как можно ближе к столбцу «Букмекер».

### Этап 3. Кластеризация (`03_clustering.py`)

1. **Профиль команды за сезон** — среднее за матч: голы (забитые/пропущенные), удары, удары в створ, угловые (свои/соперника), жёлтые и красные карточки. Всего 200 профилей (20 команд × 10 сезонов).
2. **Стандартизация** (`StandardScaler`): удары (~12 за матч) и красные карточки (~0.05) приводятся к одному масштабу, иначе удары «перевесят» всё остальное.
3. **K-means** обучается **только на сезонах train**, а остальные сезоны лишь распределяются по готовым кластерам.
4. **Выбор k** (`figures/01_choose_k.png`): метод локтя и коэффициент силуэта для k = 2…8. Силуэт максимален при k = 2, но это просто «топ-клубы против остальных». Выбрано **k = 4**: после него кривая локтя выпрямляется, а группы ещё понятно назвать.
5. **Названия** присвоены по разнице голов центра кластера:

| Кластер | Голы за / против (за матч) | Удары за / против | Команд-сезонов |
|---|---|---|---|
| Dominant | 2.19 / 0.93 | 16.6 / 8.3 | 30 |
| Strong | 1.60 / 1.18 | 13.7 / 11.4 | 48 |
| Mid-table | 1.18 / 1.45 | 11.4 / 13.3 | 71 |
| Struggling | 0.99 / 1.75 | 10.5 / 15.0 | 51 |

Графики: `02_clusters_pca.png` (кластеры на плоскости после PCA, первая компонента объясняет 68% разброса), `03_cluster_profiles.png` (тепловая карта центров).

**Честный вывод:** кластеры делят команды прежде всего **по силе**, а не по стилю игры. Почти все признаки связаны с тем, насколько команда сильна: сильная бьёт больше, пропускает меньше, получает меньше карточек. Это нормальный результат, и его стоит упомянуть в отчёте.

**Как кластеры попадают в модель** (`model_dataset.csv`): матчу присваивается кластер команды из **прошлого** сезона (`home_prev_cluster`, `away_prev_cluster`), потому что профиль текущего сезона включает ещё не сыгранные матчи. Для всего сезона 2016/17 и для команд, вышедших из низшей лиги, ставится `-1` («Promoted / unknown»).

Уже видно, что кластер имеет смысл: дома команды из Dominant выигрывают в 69% случаев, из Strong в 53%, из Mid-table и Struggling в 36–37%.

### Что дальше (этапы 4–5)

- Взять `model_dataset.csv`, удалить строки с пустой формой.
- Логистическая регрессия и нейросеть (MLP) на одинаковых признаках, обучение на `train`, подбор параметров на `valid`.
- Эксперимент: модель с кластерами и без них.
- Сравнить с букмекером на `test` по log loss и accuracy.
- Никогда не подавать в модель колонки текущего матча: `home_goals`, `away_goals`, удары, карточки, `result`, а также `bk_prob_*` (это эталон, а не признак).

---

## English

### Project structure

```
logic vs bookmaker/
├── data/
│   ├── epl_raw.csv               raw data (football-data.co.uk)
│   ├── matches.csv               stage 1: cleaned matches
│   ├── features.csv              stage 2: features + bookmaker probabilities + split
│   ├── team_season_clusters.csv  stage 3: cluster of every team in every season
│   └── model_dataset.csv         final modelling table (stage 2 + stage 3)
├── figures/                      clustering plots
├── src/
│   ├── 01_prepare_data.py
│   ├── 02_features.py
│   └── 03_clustering.py
└── requirements.txt
```

### How to run

```
pip install -r requirements.txt
python src/01_prepare_data.py
python src/02_features.py
python src/03_clustering.py
```

Run the scripts in order. Stages 2 and 3 only depend on `matches.csv`, so they can be developed in parallel.

### Data

- 10 Premier League seasons (2016/17 – 2025/26), 3,800 matches, 20 teams per season.
- Per match: teams, goals, result, shots, shots on target, corners, cards and betting odds.
- **The odds come from a single bookmaker: Bet365.** They are not a market average or median.
- Match result (`result`): `H` home win, `D` draw, `A` away win. This is the prediction target.

Outcome shares: home 44.6%, away 32.1%, draw 23.2%.

### Stage 1. Data preparation (`01_prepare_data.py`)

1. Renames columns to readable names (`FTHG` → `home_goals`, etc.).
2. Checks for missing values, duplicates, valid results and valid odds (odds must be > 1).
3. Sorts by date and gives every match an id (`match_id`).

Check result: 0 missing values, 0 duplicates, 380 matches in every season.

### Stage 2. Features (`02_features.py`)

**Key rule: a match may only use information that was known BEFORE kick-off.**
If a feature contains the result of the match itself (or of later matches), the model looks great on paper but cannot work in reality. This is called data leakage.

**2.1. Team form** — average over the team's last 5 matches before the current one:

| Feature | Meaning |
|---|---|
| `form_goals_for` | goals scored |
| `form_goals_against` | goals conceded |
| `form_sot_for` / `form_sot_against` | shots on target (own / opponent) |
| `form_points` | points (win 3, draw 1, loss 0) |

Each feature exists for the home team (`home_form_...`), the away team (`away_form_...`) and as a difference (`diff_form_...`).
At least 3 previous matches are required, otherwise the value is empty. This affects 71 matches at the very start of the data; drop them before training.

**2.2. Elo rating** — team strength as one number, updated after every match:

```
home expectation  E = 1 / (1 + 10^(-(R_home + 65 - R_away) / 400))
actual result     S = 1 (win), 0.5 (draw), 0 (loss)
change            Δ = 20 × G × (S − E)
R_home += Δ,      R_away −= Δ
```

- 65 is the home advantage, 20 is the update speed (K).
- G is the goal-difference multiplier: 1 (margin 0–1), 1.5 (margin 2), (11 + N)/8 (margin N ≥ 3).
- Start at 1500. Before each new season ratings move 25% back towards 1500 (squads change).
- Newly promoted teams start at 1450.
- The table stores the rating **before** the match; the update happens afterwards.

Columns: `elo_home`, `elo_away`, `elo_diff` (difference incl. home advantage), `elo_expected_home`.
Sanity check: `elo_expected_home` has a 0.93 correlation with the bookmaker's home probability, so the rating captures real team strength.

**2.3. Bookmaker probabilities** — the benchmark our models are compared against:

```
q = 1 / odds                         for each outcome
margin = q_H + q_D + q_A − 1         (4.7% on average)
probability = q / (q_H + q_D + q_A)  (sums to 1)
```

Columns: `bk_prob_home`, `bk_prob_draw`, `bk_prob_away`, `bookmaker_margin`.

**2.4. Time-based split** (column `split`):

| Part | Seasons | Matches | Purpose |
|---|---|---|---|
| `train` | 2016/17 – 2022/23 | 2,660 | model training |
| `valid` | 2023/24 | 380 | parameter tuning |
| `test` | 2024/25 – 2025/26 | 760 | final evaluation, used once at the end |

A random split is not allowed here: the model would "see the future".
`target` is the result as a number: 0 = H, 1 = D, 2 = A.

**Baselines** (log loss: lower is better; accuracy: share of correctly predicted outcomes):

| | Bookmaker log loss | Bookmaker accuracy | "Always home" accuracy | Uniform 1/3 log loss |
|---|---|---|---|---|
| train | 0.947 | 56.2% | 45.3% | 1.099 |
| valid | 0.909 | 59.7% | 46.1% | 1.099 |
| test | 0.995 | 51.4% | 41.7% | 1.099 |

The goal of our models is to get as close as possible to the "Bookmaker" column.

### Stage 3. Clustering (`03_clustering.py`)

1. **Team-season profile** — per-match averages: goals (scored/conceded), shots, shots on target, corners (own/opponent), yellow and red cards. 200 profiles in total (20 teams × 10 seasons).
2. **Standardisation** (`StandardScaler`): shots (~12 per match) and red cards (~0.05) are put on the same scale, otherwise shots would dominate everything else.
3. **K-means** is fitted **on train seasons only**; other seasons are only assigned to the existing clusters.
4. **Choosing k** (`figures/01_choose_k.png`): elbow method and silhouette score for k = 2…8. Silhouette peaks at k = 2, but that is just "top clubs vs the rest". We chose **k = 4**: the elbow curve flattens after it and the groups are still easy to name.
5. **Names** are assigned by the goal difference of each cluster centre:

| Cluster | Goals for / against (per match) | Shots for / against | Team-seasons |
|---|---|---|---|
| Dominant | 2.19 / 0.93 | 16.6 / 8.3 | 30 |
| Strong | 1.60 / 1.18 | 13.7 / 11.4 | 48 |
| Mid-table | 1.18 / 1.45 | 11.4 / 13.3 | 71 |
| Struggling | 0.99 / 1.75 | 10.5 / 15.0 | 51 |

Plots: `02_clusters_pca.png` (clusters on a 2D PCA plane; the first component explains 68% of the variance), `03_cluster_profiles.png` (heatmap of cluster centres).

**Honest conclusion:** the clusters separate teams mainly **by strength**, not by playing style. Almost every feature is tied to how good a team is: strong teams shoot more, concede less and get fewer cards. This is a valid result and worth mentioning in the report.

**How clusters enter the model** (`model_dataset.csv`): each match gets the team's cluster from the **previous** season (`home_prev_cluster`, `away_prev_cluster`), because the current-season profile includes matches that have not been played yet. The whole 2016/17 season and newly promoted teams get `-1` ("Promoted / unknown").

The clusters already carry signal: at home, Dominant teams win 69% of matches, Strong 53%, Mid-table and Struggling 36–37%.

### Next steps (stages 4–5)

- Load `model_dataset.csv` and drop rows with empty form values.
- Logistic regression and a neural network (MLP) on the same features; train on `train`, tune on `valid`.
- Experiment: model with clusters vs without clusters.
- Compare with the bookmaker on `test` using log loss and accuracy.
- Never feed the model columns from the current match: `home_goals`, `away_goals`, shots, cards, `result`, nor `bk_prob_*` (that is the benchmark, not a feature).
