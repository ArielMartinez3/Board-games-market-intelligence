"""Genera las figuras principales del proyecto para documentación y portfolio."""

from pathlib import Path

import pandas as pd

from modeling import evaluar_modelos
from text_processing import extraer_topicos_nmf, limpiar_texto_descripcion
from visualization import (
    graficar_complejidad_adopcion,
    graficar_evolucion_mercado,
    graficar_mensaje_ejecutivo,
    graficar_matriz_oportunidad,
    graficar_topicos_vs_rating,
)


OUTPUT_DIR = Path("docs/figures")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("juegos_limpios.csv")

    df["desc_limpia"] = df["description"].apply(limpiar_texto_descripcion)
    _, _, topic_matrix, _ = extraer_topicos_nmf(
        df["desc_limpia"].tolist(),
        n_topicos=5,
        max_features=2500,
    )
    df["topico_dominante"] = topic_matrix.argmax(axis=1)
    df["topico_nombre"] = df["topico_dominante"].map(
        {
            0: "Gestión urbana y construcción",
            1: "Estrategia militar y guerra",
            2: "Cartas, rondas y objetivos",
            3: "Dados y azar",
            4: "Losetas y puntuación",
        }
    )

    figures = [
        (
            graficar_topicos_vs_rating(
                df, output_path=str(OUTPUT_DIR / "topics-vs-rating.png")
            ),
            "topics-vs-rating.png",
        ),
        (
            graficar_complejidad_adopcion(
                df, output_path=str(OUTPUT_DIR / "complexity-vs-adoption.png")
            ),
            "complexity-vs-adoption.png",
        ),
        (
            graficar_evolucion_mercado(
                df[df["year_consolidated"].between(1990, 2021)],
                output_path=str(OUTPUT_DIR / "market-evolution.png"),
            ),
            "market-evolution.png",
        ),
    ]
    df["tasa_deseo"] = df["wishing"] / (df["owned"] + 1)
    figures.append(
        (
            graficar_matriz_oportunidad(
                df, output_path=str(OUTPUT_DIR / "editorial-opportunity.png")
            ),
            "editorial-opportunity.png",
        )
    )

    results = evaluar_modelos(df)
    results["roc_figure"].savefig(
        OUTPUT_DIR / "model-performance.png", bbox_inches="tight"
    )
    results["importance_figure"].savefig(
        OUTPUT_DIR / "feature-importance.png", bbox_inches="tight"
    )
    graficar_mensaje_ejecutivo(
        r2_value=float(
            results["regression"].loc[
                results["regression"]["Modelo"] == "Gradient boosting", "R2"
            ].iloc[0]
        ),
        output_path=str(OUTPUT_DIR / "executive-takeaway.png"),
    ).clear()
    for figure, _ in figures:
        figure.clear()
    print(f"Generated {len(figures) + 3} figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
