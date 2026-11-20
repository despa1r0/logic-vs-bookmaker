from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "matches.csv"
FEATURES = ROOT / "data" / "features.csv"
OUT_CLUSTERS = ROOT / "data" / "team_season_clusters.csv"
OUT_MODEL = ROOT / "data" / "model_dataset.csv"
FIG = ROOT / "figures"

TRAIN_SEASONS = ["2016/17", "2017/18", "2018/19", "2019/20", "2020/21", "2021/22", "2022/23"]
K_RANGE = range(2, 9)
N_CLUSTERS = 4
RANDOM_STATE = 42

PROFILE = [
    "goals_for", "goals_against", "shots_for", "shots_against",
    "sot_for", "sot_against", "corners_for", "corners_against",
    "yellow_cards", "red_cards",
]


def team_season_profiles(df: pd.DataFrame) -> pd.DataFrame:
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        sides.append(pd.DataFrame({
            "season": df["season"], "team": df[f"{side}_team"],
            "goals_for": df[f"{side}_goals"], "goals_against": df[f"{other}_goals"],
            "shots_for": df[f"{side}_shots"], "shots_against": df[f"{other}_shots"],
            "sot_for": df[f"{side}_shots_on_target"], "sot_against": df[f"{other}_shots_on_target"],
            "corners_for": df[f"{side}_corners"], "corners_against": df[f"{other}_corners"],
            "yellow_cards": df[f"{side}_yellow_cards"], "red_cards": df[f"{side}_red_cards"],
        }))
    return pd.concat(sides).groupby(["season", "team"], as_index=False)[PROFILE].mean()


def choose_k(x_train: np.ndarray) -> None:
    inertia, silhouette = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit(x_train)
        inertia.append(km.inertia_)
        silhouette.append(silhouette_score(x_train, km.labels_))
        print(f"  k={k}: inertia={km.inertia_:.1f} silhouette={silhouette[-1]:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(list(K_RANGE), inertia, marker="o")
    ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
    ax[1].plot(list(K_RANGE), silhouette, marker="o", color="tab:orange")
    ax[1].set(title="Silhouette score", xlabel="k", ylabel="Silhouette")
    for a in ax:
        a.axvline(N_CLUSTERS, color="grey", linestyle="--", linewidth=1)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "01_choose_k.png", dpi=150)
    plt.close(fig)


def name_clusters(centers: pd.DataFrame) -> dict[int, str]:
    """Rank clusters by goal difference per match and name them."""
    strength = (centers["goals_for"] - centers["goals_against"]).sort_values(ascending=False)
    names = ["Dominant", "Strong", "Mid-table", "Struggling"]
    if len(strength) != len(names):
        names = [f"Tier {i + 1}" for i in range(len(strength))]
    return {cluster: names[rank] for rank, cluster in enumerate(strength.index)}


def plot_clusters(profiles: pd.DataFrame, x_all: np.ndarray, pca: PCA) -> None:
    pts = pca.transform(x_all)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, grp in profiles.groupby("cluster_name"):
        idx = grp.index.to_numpy()
        ax.scatter(pts[idx, 0], pts[idx, 1], label=name, alpha=0.7, s=35)
    for team in ["Man City", "Liverpool", "Arsenal", "Burnley", "Sheffield United"]:
        sel = profiles[(profiles["team"] == team)].tail(1)
        for i in sel.index:
            ax.annotate(f"{team} {profiles.at[i, 'season']}", pts[i, :2], fontsize=8, alpha=0.8)
    ev = pca.explained_variance_ratio_
    ax.set(title="Team-season clusters (PCA projection)",
           xlabel=f"PC1 ({ev[0]:.0%} variance)", ylabel=f"PC2 ({ev[1]:.0%} variance)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "02_clusters_pca.png", dpi=150)
    plt.close(fig)


def plot_profiles(centers_z: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(centers_z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(centers_z.columns)), centers_z.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(centers_z.index)), centers_z.index)
    for i in range(centers_z.shape[0]):
        for j in range(centers_z.shape[1]):
            ax.text(j, i, f"{centers_z.iat[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Cluster centers (standardized, 0 = league average)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "03_cluster_profiles.png", dpi=150)
    plt.close(fig)


def attach_to_matches(clusters: pd.DataFrame) -> None:
    if not FEATURES.exists():
        print("features.csv not found, skipping model_dataset.csv")
        return
    seasons = sorted(clusters["season"].unique())
    prev = clusters[["season", "team", "cluster", "cluster_name"]].copy()
    # cluster from the PREVIOUS season, so the current season result is never used
    prev["season"] = prev["season"].map(dict(zip(seasons[:-1], seasons[1:])))
    prev = prev.dropna(subset=["season"])

    df = pd.read_csv(FEATURES)
    for side in ("home", "away"):
        part = prev.rename(columns={
            "team": f"{side}_team",
            "cluster": f"{side}_prev_cluster",
            "cluster_name": f"{side}_prev_cluster_name",
        })
        df = df.merge(part, on=["season", f"{side}_team"], how="left")
        df[f"{side}_prev_cluster"] = df[f"{side}_prev_cluster"].fillna(-1).astype(int)
        df[f"{side}_prev_cluster_name"] = df[f"{side}_prev_cluster_name"].fillna("Promoted / unknown")
    df.to_csv(OUT_MODEL, index=False)
    print(f"Saved {OUT_MODEL.name}: {df.shape}")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    profiles = team_season_profiles(pd.read_csv(IN))
    is_train = profiles["season"].isin(TRAIN_SEASONS)

    scaler = StandardScaler().fit(profiles.loc[is_train, PROFILE])
    x_all = scaler.transform(profiles[PROFILE])
    x_train = x_all[is_train.to_numpy()]

    print(f"Team-seasons: {len(profiles)} (train: {is_train.sum()})")
    print("Choosing k:")
    choose_k(x_train)

    km = KMeans(n_clusters=N_CLUSTERS, n_init=50, random_state=RANDOM_STATE).fit(x_train)
    profiles["cluster"] = km.predict(x_all)

    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=PROFILE)
    names = name_clusters(centers)
    profiles["cluster_name"] = profiles["cluster"].map(names)

    centers.index = [names[i] for i in centers.index]
    order = [n for n in ["Dominant", "Strong", "Mid-table", "Struggling"] if n in centers.index]
    centers = centers.loc[order or centers.index]
    centers_z = pd.DataFrame(km.cluster_centers_, columns=PROFILE, index=[names[i] for i in range(N_CLUSTERS)])
    centers_z = centers_z.loc[centers.index]

    print("\nCluster centers (per match):")
    print(centers.round(2).to_string())
    print("\nTeam-seasons per cluster:", profiles["cluster_name"].value_counts().to_dict())

    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(x_train)
    plot_clusters(profiles, x_all, pca)
    plot_profiles(centers_z)

    profiles.round(3).to_csv(OUT_CLUSTERS, index=False)
    print(f"Saved {OUT_CLUSTERS.name}: {profiles.shape}")
    attach_to_matches(profiles)


if __name__ == "__main__":
    main()
