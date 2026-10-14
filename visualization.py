"""Visualizaciones consistentes para comunicar los resultados del análisis."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns


PALETTE = {
    "ink": "#243447",
    "muted": "#5B6B7A",
    "blue": "#2F6690",
    "teal": "#3C8D8A",
    "orange": "#D9822B",
    "green": "#3A7D44",
    "red": "#B54747",
    "sand": "#F4EFE6",
    "light_blue": "#E7F0F7",
    "light_teal": "#E7F3F1",
    "light_orange": "#FFF1DE",
}

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "board_games_market",
    [PALETTE["sand"], PALETTE["teal"], PALETTE["blue"]],
)


def configurar_estilo() -> None:
    """Aplica un estilo sobrio, legible y consistente a todas las figuras."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlepad": 14,
            "axes.labelcolor": PALETTE["ink"],
            "axes.edgecolor": "#C8D0D8",
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _guardar_si_corresponde(fig: plt.Figure, output_path: Optional[str]) -> None:
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")


def graficar_topicos_vs_rating(
    df: pd.DataFrame,
    topic_col: str = "topico_nombre",
    rating_col: str = "bayes_average",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Compara la distribución de rating por tópico, ordenada por mediana."""
    configurar_estilo()
    orden = (
        df.groupby(topic_col)[rating_col]
        .median()
        .sort_values(ascending=False)
        .index
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.boxplot(
        data=df,
        x=rating_col,
        y=topic_col,
        order=orden,
        color=PALETTE["blue"],
        saturation=0.8,
        width=0.62,
        fliersize=2,
        ax=ax,
    )
    global_median = df[rating_col].median()
    ax.axvline(
        global_median,
        color=PALETTE["orange"],
        linestyle="--",
        linewidth=1.5,
        label=f"Mediana del mercado: {global_median:.2f}",
    )
    ax.set_title("¿Qué temas se asocian con una mejor recepción crítica?")
    ax.set_xlabel("Calificación bayesiana, escala 1 a 10")
    ax.set_ylabel("")
    ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    _guardar_si_corresponde(fig, output_path)
    return fig


def graficar_complejidad_adopcion(
    df: pd.DataFrame,
    mechanics_col: str = "num_mecanicas",
    duration_col: str = "playingtime",
    adoption_col: str = "owned",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Muestra el trade-off entre densidad, duración y adopción comunitaria."""
    configurar_estilo()
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    points = ax.scatter(
        df[mechanics_col],
        df[duration_col],
        c=df[adoption_col],
        cmap=SEQUENTIAL_CMAP,
        alpha=0.58,
        s=28,
        linewidth=0,
    )
    ax.set_yscale("log")
    colorbar = fig.colorbar(points, ax=ax, pad=0.02)
    colorbar.set_label("Copias registradas en colecciones (escala logarítmica)")
    ax.set_title("La profundidad del diseño compite con el tiempo disponible")
    ax.set_xlabel("Cantidad de mecánicas")
    ax.set_ylabel("Duración estimada, minutos (escala logarítmica)")
    ax.text(
        0.01,
        -0.23,
        "Cada punto representa un juego. La escala logarítmica evita que los títulos extremos oculten al resto.",
        transform=ax.transAxes,
        color=PALETTE["muted"],
        fontsize=9,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    _guardar_si_corresponde(fig, output_path)
    return fig


def graficar_evolucion_mercado(
    df: pd.DataFrame,
    year_col: str = "year_consolidated",
    rating_col: str = "bayes_average",
    mechanics_col: str = "num_mecanicas",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Contrasta la evolución anual del rating y la densidad mecánica."""
    configurar_estilo()
    evolution = (
        df.groupby(year_col)
        .agg(
            rating_medio=(rating_col, "mean"),
            mecanicas_medias=(mechanics_col, "mean"),
        )
        .reset_index()
    )
    fig, ax_rating = plt.subplots(figsize=(10, 5.5))
    ax_mechanics = ax_rating.twinx()
    ax_rating.plot(
        evolution[year_col],
        evolution["rating_medio"],
        color=PALETTE["blue"],
        linewidth=2.4,
        label="Rating medio",
    )
    ax_mechanics.plot(
        evolution[year_col],
        evolution["mecanicas_medias"],
        color=PALETTE["orange"],
        linewidth=2.4,
        linestyle="--",
        label="Mecánicas medias",
    )
    ax_rating.set_title("El mercado combina una valoración creciente con diseños más densos")
    ax_rating.set_xlabel("Año de publicación")
    ax_rating.set_ylabel("Rating bayesiano medio", color=PALETTE["blue"])
    ax_mechanics.set_ylabel("Mecánicas medias por juego", color=PALETTE["orange"])
    lines = ax_rating.lines + ax_mechanics.lines
    ax_rating.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper left")
    ax_rating.grid(axis="x", visible=False)
    sns.despine(ax=ax_rating, right=False)
    fig.tight_layout()
    _guardar_si_corresponde(fig, output_path)
    return fig


def graficar_matriz_oportunidad(
    df: pd.DataFrame,
    rating_col: str = "bayes_average",
    demand_col: str = "tasa_deseo",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Ubica juegos por calidad crítica y demanda relativa no satisfecha."""
    configurar_estilo()
    working = df[[rating_col, demand_col]].dropna().copy()
    x_cut = working[rating_col].median()
    y_cut = working[demand_col].median()
    y_max = working[demand_col].quantile(0.99)
    x_min = working[rating_col].min() - 0.2
    x_max = working[rating_col].max() + 0.2

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axvspan(x_cut, x_max, y_cut / y_max, 1, color=PALETTE["light_teal"], alpha=0.9)
    ax.axvspan(x_cut, x_max, 0, y_cut / y_max, color=PALETTE["light_blue"], alpha=0.9)
    ax.axvspan(x_min, x_cut, y_cut / y_max, 1, color=PALETTE["light_orange"], alpha=0.9)
    ax.scatter(
        working[rating_col],
        working[demand_col],
        color=PALETTE["ink"],
        alpha=0.28,
        s=20,
        linewidth=0,
    )
    ax.axvline(x_cut, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.axhline(y_cut, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.text(x_max, y_max * 0.94, "OPORTUNIDAD\ncalidad + demanda", ha="right", va="top", color=PALETTE["green"], weight="bold")
    ax.text(x_max, y_max * 0.06, "CATÁLOGO CONSOLIDADO", ha="right", va="bottom", color=PALETTE["blue"], weight="bold")
    ax.text(x_min, y_max * 0.94, "HYPE A VALIDAR", ha="left", va="top", color=PALETTE["orange"], weight="bold")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    ax.set_title("Matriz editorial: calidad crítica frente a demanda insatisfecha")
    ax.set_xlabel(f"Rating bayesiano (mediana: {x_cut:.2f})")
    ax.set_ylabel(f"Usuarios que desean / usuarios que poseen (mediana: {y_cut:.2f})")
    ax.text(
        0,
        -0.18,
        "La matriz prioriza investigación comercial; no reemplaza validar precio, costos, derechos y distribución.",
        transform=ax.transAxes,
        color=PALETTE["muted"],
        fontsize=9,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    _guardar_si_corresponde(fig, output_path)
    return fig


def graficar_mensaje_ejecutivo(
    r2_value: float = 0.276,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Resume para público general cuánto explica el diseño según el modelo."""
    configurar_estilo()
    explained = max(0.0, min(float(r2_value), 1.0))
    unexplained = 1 - explained
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.barh(
        [0],
        [explained],
        color=PALETTE["teal"],
        height=0.42,
        label="Señal explicada por el diseño",
    )
    ax.barh(
        [0],
        [unexplained],
        left=[explained],
        color=PALETTE["light_blue"],
        edgecolor=PALETTE["blue"],
        linewidth=0.8,
        height=0.42,
        label="Factores que el dataset no observa",
    )
    ax.text(explained / 2, 0, f"{explained:.1%}", ha="center", va="center", weight="bold", color="white")
    ax.text(explained + unexplained / 2, 0, f"{unexplained:.1%}", ha="center", va="center", weight="bold", color=PALETTE["ink"])
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("El diseño aporta una señal, pero no explica todo el éxito")
    ax.set_xlabel("Parte de la variación de la valoración crítica explicada en el test")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    _guardar_si_corresponde(fig, output_path)
    return fig
