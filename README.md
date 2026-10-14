# Board Games Market Intelligence

## From board-game metadata to editorial decisions

An end-to-end data science project that studies the BoardGameGeek catalog to answer a practical question for a publisher:

> **Which product characteristics are associated with critical reception, community adoption and editorial opportunity before production begins?**

The project combines data integration and quality control, exploratory analysis, natural language processing, product segmentation and predictive modeling. The main deliverable is the notebook [Trabajo_Practico_Integrador_Ariel_Martinez.ipynb](Trabajo_Practico_Integrador_Ariel_Martinez.ipynb), supported by reusable Python modules for cleaning and text processing.

## Executive summary

The analysis uses BoardGameGeek data distributed through the [TidyTuesday dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2022/2022-01-25). After joining the ratings and details sources, the workflow applies a minimum threshold of 30 ratings, repairs inconsistent player and duration fields, and creates design-density features from mechanics and categories.

The notebook investigates six editorial questions:

- Which semantic topics emerge from game descriptions through TF-IDF + NMF?
- How do mechanical density and playing time relate to community adoption?
- Has design complexity evolved together with critical ratings over time?
- Where do critical quality and latent demand point to licensing or reprint opportunities?
- How can games be grouped into product archetypes using intrinsic design attributes?
- How much can pre-launch design attributes explain critical reception and top-tier performance?

### Key findings reported in the notebook

| Area | Result | Editorial reading |
| --- | --- | --- |
| Design density | Moderate positive association between number of mechanics and Bayesian rating (`r = 0.38`) | Deeper systems may improve specialist appeal, but they do not guarantee mass adoption |
| Duration and adoption | The notebook reports a drop in `owned` volume beyond roughly 90 minutes | Long games can be attractive to a niche while carrying a narrower retail audience |
| Topic modeling | Economic/engine-building and deduction/hidden-role descriptions rank above family/dice/ luck topics in reported median ratings | Theme and mechanism positioning can help define differentiated catalog bets |
| Market opportunity | The reported quality-versus-demand matrix identifies a high-quality, under-owned segment | A useful screen for localization, licensing or reprint candidates |
| Predictive ceiling | `HistGradientBoostingRegressor`: reported $R^2 = 0.224$; `RandomForestClassifier`: reported ROC-AUC $= 0.771$ | Design attributes provide a useful prior, not a commercial forecast |

These figures are descriptive results from the notebook's analysis and should be revalidated when the source data or preprocessing choices change.

## Methodology

### Data preparation

1. Download `ratings.csv` and `details.csv` from the TidyTuesday source.
2. Keep records present in both sources through an inner join.
3. Consolidate publication year using the earliest valid value.
4. Repair inverted player ranges, missing player counts and invalid durations.
5. Exclude durations above 720 minutes from the commercial analysis.
6. Keep games with at least 30 ratings to reduce extreme small-sample effects.
7. Parse list-like metadata and derive the number of mechanics and categories.

The cleaning decisions are documented in the notebook and implemented in [data_cleaning.py](data_cleaning.py).

### Analysis layers

- **NLP:** HTML cleaning, TF-IDF with domain stopwords and NMF topic extraction.
- **Exploration:** temporal trends, adoption patterns and a critical-quality versus latent-demand matrix.
- **Segmentation:** standardized design attributes clustered with K-Means and visualized through PCA.
- **Regression:** baseline mean, Ridge and HistGradientBoosting models for `bayes_average`.
- **Classification:** stratified baseline, Logistic Regression and Random Forest for the top 15% of Bayesian ratings.
- **Leakage control:** the predictive models use only pre-launch design attributes; post-launch variables such as `owned`, `users_rated`, `wishing`, `wanting`, `trading` and `rank` are excluded from the feature matrix.

## Repository map

```text
.
├── Trabajo_Practico_Integrador_Ariel_Martinez.ipynb  # Full analysis and narrative
├── data_cleaning.py                                  # Download, integration and cleaning
├── text_processing.py                                # Text normalization and NMF helper
├── requirements.txt                                  # Python dependencies
├── README.md                                          # Project documentation
└── .gitignore                                         # Local and generated files excluded from Git
```

Generated datasets and notebook intermediates are intentionally not committed by default. The workflow writes the cleaned dataset to `juegos_limpios.csv` when the cleaning module is executed.

## Quickstart

### 1. Create an environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The notebook also requires the NLTK English stopwords corpus. Run this once after installation:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### 3. Generate the cleaned dataset

```bash
python data_cleaning.py
```

### 4. Run the notebook

Open [Trabajo_Practico_Integrador_Ariel_Martinez.ipynb](Trabajo_Practico_Integrador_Ariel_Martinez.ipynb) in Jupyter or VS Code and execute the cells in order. The notebook expects `juegos_limpios.csv` in the project root, so run the cleaning step first.

## Responsible interpretation

This is a decision-support analysis, not a guarantee of sales. BoardGameGeek is a self-selected hobbyist community and the snapshot used by the source data does not represent every retail audience. The dataset also lacks important commercial variables such as price, production quality, component cost, marketing reach, distribution and artwork.

The predictive results should therefore be read as a **design prior**: useful for screening and comparison, insufficient for approving a production run on its own.

## Data source and attribution

The data comes from the [TidyTuesday BoardGameGeek dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2022/2022-01-25), which in turn documents the underlying BoardGameGeek source. Please review the original source terms before redistributing data files.

## Author

**Ariel Martinez**  
Data analysis, data cleaning, NLP, segmentation and predictive modeling.