"""Módulo para el procesamiento de texto (NLP) y extracción de tópicos de las descripciones."""

import html
import re
from typing import Any, Dict, List, Tuple
from nltk.corpus import stopwords
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def limpiar_texto_descripcion(texto: object) -> str:
  """Limpia entidades HTML, tags, caracteres no alfanuméricos y normaliza texto."""
  if not isinstance(texto, str) or pd.isna(texto):
    return ""

  # 1. Decodificar entidades HTML (&amp;, &#10;, &quot;)
  t = html.unescape(texto)

  # 2. Remover etiquetas HTML (<br/>, <p>, etc.)
  t = re.sub(r"<[^>]+>", " ", t)

  # 3. Remover secuencias de escape y caracteres especiales
  t = re.sub(r"[\r\n\t]", " ", t)
  t = re.sub(r"[^a-zA-Z\s]", " ", t)

  # 4. Normalizar espacios en blanco y minúsculas
  t = re.sub(r"\s+", " ", t).strip().lower()
  return t


def extraer_topicos_nmf(
  textos: List[str],
  n_topicos: int = 5,
  max_features: int = 3000,
  random_state: int = 42,
) -> Tuple[NMF, TfidfVectorizer, Any, Dict[str, List[str]]]:
  """Aplica TF-IDF + NMF (Non-negative Matrix Factorization) para descubrir tópicos temáticos.

  NMF resulta más veloz y genera tópicos más cohesivos y diferenciados que LDA
  en descripciones breves.
  """
  try:
    english_stopwords = list(stopwords.words("english"))
  except LookupError as err:
    raise RuntimeError(
        "Falta el corpus de stopwords de NLTK. Ejecuta "
        "nltk.download('stopwords') antes del análisis."
    ) from err

  # Stopwords en inglés extendidas con términos genéricos del dominio de juegos.
  custom_stopwords = english_stopwords + [
      "game",
      "games",
      "player",
      "players",
      "play",
      "played",
      "playing",
      "board",
      "card",
      "cards",
      "turn",
      "points",
      "rule",
      "rules",
      "win",
  ]

  vectorizer = TfidfVectorizer(
      max_df=0.85,
      min_df=5,
      max_features=max_features,
      stop_words=custom_stopwords,
      ngram_range=(1, 2),
  )
  matriz_tfidf = vectorizer.fit_transform(textos)

  nmf_model = NMF(
      n_components=n_topicos,
      random_state=random_state,
      init="nndsvda",
      max_iter=500,
  )
  topicos_weights = nmf_model.fit_transform(matriz_tfidf)

  nombres_palabras = vectorizer.get_feature_names_out()
  topicos_vocabulario = {}

  for topic_idx, topic in enumerate(nmf_model.components_):
    top_words_idx = topic.argsort()[::-1][:10]
    top_words = [nombres_palabras[i] for i in top_words_idx]
    topicos_vocabulario[f"Topico_{topic_idx}"] = top_words

  return nmf_model, vectorizer, topicos_weights, topicos_vocabulario