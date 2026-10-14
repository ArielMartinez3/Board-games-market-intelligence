"""Segmentación de perfiles de producto a partir de atributos de diseño."""

from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from visualization import PALETTE, configurar_estilo


CLUSTER_FEATURES = [
    "minplayers",
    "maxplayers",
    "playingtime",
    "minage",
    "num_mecanicas",
    "num_categorias",
]


PROFILE_NAMES = {
    0: "Gateway / familiares",
    1: "Eurogames estratégicos",
    2: "Party games masivos",
    3: "Duelos tácticos",
}


def segmentar_perfiles(
    df: pd.DataFrame,
    cluster_features: Iterable[str] = CLUSTER_FEATURES,
    k_values: Iterable[int] = range(2, 8),
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
    """Selecciona k por silhouette score y devuelve perfiles, métricas y figura PCA."""
    configurar_estilo()
    features = list(cluster_features)
    working = df.copy()
    matrix = working[features].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0)
    scaled = StandardScaler().fit_transform(matrix)

    metrics = []
    models: Dict[int, KMeans] = {}
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(scaled)
        metrics.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(scaled, labels),
            }
        )
        models[k] = model

    metrics_df = pd.DataFrame(metrics)
    selected_k = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])
    selected_model = models[selected_k]
    result = working.copy()
    result["cluster_perfil"] = selected_model.labels_

    pca = PCA(n_components=2, random_state=random_state)
    coordinates = pca.fit_transform(scaled)
    result["pca1"] = coordinates[:, 0]
    result["pca2"] = coordinates[:, 1]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(metrics_df["k"], metrics_df["inertia"], marker="o", color=PALETTE["blue"])
    axes[0].set_title("Codo: complejidad interna")
    axes[0].set_xlabel("Número de clusters")
    axes[0].set_ylabel("Inercia")
    axes[1].plot(metrics_df["k"], metrics_df["silhouette"], marker="o", color=PALETTE["orange"])
    axes[1].axvline(selected_k, linestyle="--", color=PALETTE["muted"])
    axes[1].set_title(f"Silhouette: k elegido = {selected_k}")
    axes[1].set_xlabel("Número de clusters")
    axes[1].set_ylabel("Silhouette score")
    figure.suptitle("Selección reproducible del número de perfiles", weight="bold")
    figure.tight_layout()

    return result, metrics_df, figure


def graficar_perfiles_pca(
    segmented_df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure:
    """Grafica los perfiles obtenidos en el espacio PCA."""
    configurar_estilo()
    colors = [
        PALETTE["blue"],
        PALETTE["teal"],
        PALETTE["orange"],
        PALETTE["red"],
    ][: segmented_df["cluster_perfil"].nunique()]
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=segmented_df,
        x="pca1",
        y="pca2",
        hue="cluster_perfil",
        palette=colors,
        alpha=0.65,
        s=32,
        linewidth=0,
        ax=axis,
    )
    axis.set_title("Perfiles de producto según atributos de diseño")
    axis.set_xlabel("Componente principal 1")
    axis.set_ylabel("Componente principal 2")
    axis.legend(title="Cluster", frameon=False)
    figure.tight_layout()
    if output_path:
        figure.savefig(output_path, bbox_inches="tight")
    return figure
