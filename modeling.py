"""Modelos de apoyo para evaluar el potencial de diseño antes del lanzamiento."""

import ast
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from visualization import PALETTE, configurar_estilo


NUMERIC_FEATURES = [
    "minplayers",
    "maxplayers",
    "playingtime",
    "minage",
    "num_mecanicas",
    "num_categorias",
]


def _parse_list_column(value: object) -> List[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if not isinstance(value, str) or pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return [item for item in parsed if isinstance(item, str)] if isinstance(parsed, list) else []


def _vectorizar_diseno(
    frame: pd.DataFrame,
    top_mecanicas: List[str],
    top_categorias: List[str],
) -> pd.DataFrame:
    result = frame[NUMERIC_FEATURES].copy()
    for column in NUMERIC_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.fillna(result.median(numeric_only=True)).fillna(0)

    for mechanic in top_mecanicas:
        column_name = f"mec_{mechanic.replace(' ', '_')[:20]}"
        result[column_name] = frame["boardgamemechanic_parsed"].apply(
            lambda values: int(mechanic in values)
        )
    for category in top_categorias:
        column_name = f"cat_{category.replace(' ', '_')[:20]}"
        result[column_name] = frame["boardgamecategory_parsed"].apply(
            lambda values: int(category in values)
        )
    return result


def evaluar_modelos(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, object]:
    """Entrena, evalúa y grafica los modelos sin utilizar variables post-lanzamiento."""
    configurar_estilo()
    working = df.copy()
    for column in ["boardgamemechanic_parsed", "boardgamecategory_parsed"]:
        working[column] = working[column].apply(_parse_list_column)

    threshold = working["bayes_average"].quantile(0.85)
    working["es_exito"] = (working["bayes_average"] >= threshold).astype(int)
    features = working[NUMERIC_FEATURES + [
        "boardgamemechanic_parsed",
        "boardgamecategory_parsed",
    ]]
    y_reg = working["bayes_average"].to_numpy()
    y_clf = working["es_exito"].to_numpy()

    x_train_raw, x_test_raw, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        features,
        y_reg,
        y_clf,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clf,
    )

    mechanics = [item for values in x_train_raw["boardgamemechanic_parsed"] for item in values]
    categories = [item for values in x_train_raw["boardgamecategory_parsed"] for item in values]
    top_mechanics = pd.Series(mechanics).value_counts().head(15).index.tolist()
    top_categories = pd.Series(categories).value_counts().head(10).index.tolist()

    x_train = _vectorizar_diseno(x_train_raw, top_mechanics, top_categories)
    x_test = _vectorizar_diseno(x_test_raw, top_mechanics, top_categories)
    scaler = StandardScaler()
    x_train[NUMERIC_FEATURES] = scaler.fit_transform(x_train[NUMERIC_FEATURES])
    x_test[NUMERIC_FEATURES] = scaler.transform(x_test[NUMERIC_FEATURES])

    regression_models = {
        "Baseline (media)": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=10.0, random_state=random_state),
        "Gradient boosting": HistGradientBoostingRegressor(
            random_state=random_state, max_iter=150
        ),
    }
    regression_results = []
    for name, model in regression_models.items():
        model.fit(x_train, y_reg_train)
        predictions = model.predict(x_test)
        regression_results.append(
            {
                "Modelo": name,
                "RMSE": np.sqrt(mean_squared_error(y_reg_test, predictions)),
                "MAE": mean_absolute_error(y_reg_test, predictions),
                "R2": r2_score(y_reg_test, predictions),
            }
        )

    classifier_models = {
        "Baseline estratificado": DummyClassifier(
            strategy="stratified", random_state=random_state
        ),
        "Regresión logística": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }
    classifier_results = []
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, model in classifier_models.items():
        model.fit(x_train, y_clf_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_clf_test, probabilities)
        precision, recall, _ = precision_recall_curve(y_clf_test, probabilities)
        classifier_results.append(
            {
                "Modelo": name,
                "ROC-AUC": roc_auc_score(y_clf_test, probabilities),
                "Average precision": average_precision_score(y_clf_test, probabilities),
            }
        )
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(recall, precision, label=name)

    positive_rate = y_clf_test.mean()
    axes[0].plot([0, 1], [0, 1], "--", color=PALETTE["muted"], linewidth=1)
    axes[1].axhline(
        positive_rate,
        linestyle="--",
        color=PALETTE["muted"],
        linewidth=1,
        label=f"Referencia: {positive_rate:.0%}",
    )
    axes[0].set_title("ROC-AUC: capacidad de ordenar casos")
    axes[0].set_xlabel("Tasa de falsos positivos")
    axes[0].set_ylabel("Tasa de verdaderos positivos")
    axes[1].set_title("Precision-Recall: utilidad con clase minoritaria")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.25)
    figure.suptitle("¿Puede el diseño anticipar qué juegos entran al top 15%?", y=1.02, weight="bold")
    figure.tight_layout()

    forest = classifier_models["Random Forest"]
    importances = pd.Series(forest.feature_importances_, index=x_train.columns)
    top_importances = importances.sort_values(ascending=False).head(12).sort_values()
    importance_figure, importance_axis = plt.subplots(figsize=(9, 5))
    top_importances.plot.barh(ax=importance_axis, color=PALETTE["teal"])
    importance_axis.set_title("Variables que más separan los perfiles en Random Forest")
    importance_axis.set_xlabel("Importancia relativa")
    importance_axis.set_ylabel("")
    importance_figure.tight_layout()

    print("Resultados de regresión")
    print(pd.DataFrame(regression_results).round(3).to_string(index=False))
    print("\nResultados de clasificación")
    print(pd.DataFrame(classifier_results).round(3).to_string(index=False))

    return {
        "regression": pd.DataFrame(regression_results),
        "classification": pd.DataFrame(classifier_results),
        "threshold": threshold,
        "roc_figure": figure,
        "importance_figure": importance_figure,
    }
