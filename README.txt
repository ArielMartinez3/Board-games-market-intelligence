# Board Games Market Intelligence & Predictive Performance

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

Análisis exploratorio, minería de texto (NLP) y modelos predictivos para evaluar el mercado de juegos de mesa y anticipar el éxito crítico de prototipos antes de su producción editorial. Proyecto desarrollado sobre los datos de [BoardGameGeek (TidyTuesday)](https://github.com/rfordatascience/tidytuesday/tree/master/data/2022/2022-01-25).

---

## Resumen Ejecutivo

Este trabajo analiza el ecosistema de juegos de mesa de BoardGameGeek para orientar las decisiones de inversión y desarrollo de una editorial antes del lanzamiento de nuevos títulos. Tras un saneamiento e integración de fuentes que resolvió inconsistencias físicas y estabilizó la representatividad muestral ($N \ge 30$), se abordaron seis preguntas de investigación mediante análisis exploratorio, modelado de tópicos textuales (NLP/NMF) y agrupamiento arquetípico (K-Means). Se evidenció una evolución hacia mecánicas más densas y una clara prima de valoración sobre temas de gestión estratégica frente al azar tradicional. Asimismo, se implementaron dos modelos predictivos bajo un protocolo estricto sin fuga de datos (*zero data leakage*), evaluando únicamente atributos de diseño previo a la producción: una regresión sobre la calificación bayesiana y un clasificador del éxito editorial (Top 15%). Los modelos revelan que las variables de diseño explican cerca del 22% de la varianza crítica ($R^2 \approx 0.22$, $\text{ROC-AUC} \approx 0.77$), demostrando que la arquitectura estructural fija un umbral de viabilidad, pero el éxito comercial depende de factores externos no capturados en los datos (arte, marketing y componentes).

---

## Estructura del Repositorio

```text
├── README.md                      <- Documentación integral del proyecto
├── requirements.txt               <- Dependencias fijadas del entorno
├── .gitignore
├── data/
│   ├── raw/                       <- ratings.csv y details.csv originales
│   └── processed/
│       └── juegos_limpios.csv     <- Dataset procesado y saneado (Bloque 4.1)
├── notebooks/
│   └── TPI_Juegos_Mesa.ipynb      <- Notebook entregable completo y reproducible
└── src/
    ├── __init__.py
    ├── data_cleaning.py           <- Pipeline de integración, validación y saneamiento
    ├── text_processing.py         <- Pipeline NLP: limpieza HTML y extracción NMF
    ├── feature_engineering.py     <- Extracción de variables de diseño pre-lanzamiento
    └── model_evaluation.py        <- Validación de modelos, curvas ROC y métricas