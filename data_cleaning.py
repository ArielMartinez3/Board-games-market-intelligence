"""Módulo de limpieza, validación e integración de datos de juegos de mesa."""

import ast
from typing import List, Tuple
import numpy as np
import pandas as pd

URL_BASE = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/"
MAX_JUGADORES_ANALISIS = 20


def descargar_datos(base_url: str = URL_BASE) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Descarga los datasets crudos desde TidyTuesday con manejo de excepciones."""
  try:
    df_ratings = pd.read_csv(base_url + "ratings.csv")
    df_details = pd.read_csv(base_url + "details.csv")
    return df_ratings, df_details
  except Exception as err:
    raise RuntimeError(f"Fallo en la descarga de datos remotos: {err}") from err


def integrar_fuentes(
    df_ratings: pd.DataFrame, df_details: pd.DataFrame
) -> pd.DataFrame:
  """Realiza el diagnóstico de correspondencia y ejecuta el Inner Join fundamentado.

  Resuelve colisiones de nombre y año.
  """
  columnas_ratings = {"id", "users_rated"}
  columnas_details = {"id", "primary"}
  faltantes_ratings = columnas_ratings.difference(df_ratings.columns)
  faltantes_details = columnas_details.difference(df_details.columns)
  if faltantes_ratings or faltantes_details:
    raise ValueError(
        "Faltan columnas requeridas para integrar las fuentes: "
        f"ratings={sorted(faltantes_ratings)}, "
        f"details={sorted(faltantes_details)}"
    )

  ids_ratings = set(df_ratings["id"])
  ids_details = set(df_details["id"])
  discrepancias = ids_ratings.symmetric_difference(ids_details)

  print(f"[ETL] Total en ratings: {len(ids_ratings)}")
  print(f"[ETL] Total en details: {len(ids_details)}")
  print(
      f"[ETL] Registros descartados por falta de correspondencia:"
      f" {len(discrepancias)}"
  )

  # Eliminamos 'num' redundante y 'name' para preservar 'primary' como canónica
  df_r = df_ratings.drop(columns=["num", "name"], errors="ignore")
  df_d = df_details.drop(columns=["num"], errors="ignore")

  df_merged = pd.merge(df_r, df_d, on="id", how="inner")
  df_merged.rename(columns={"primary": "name"}, inplace=True)

  # Consolidación de año: priorizamos la fecha de publicación más antigua
  year_r = pd.to_numeric(df_merged["year"], errors="coerce")
  year_d = pd.to_numeric(df_merged["yearpublished"], errors="coerce")

  # Mínimo entre ambas fuentes ignorando nulos o ceros
  anios_validos = np.where(
      (year_r > 0) & (year_d > 0),
      np.minimum(year_r, year_d),
      np.where(year_r > 0, year_r, year_d),
  )
  df_merged["year_consolidated"] = anios_validos
  df_merged.drop(columns=["year", "yearpublished"], inplace=True)

  return df_merged


def parsear_lista_segura(valor: object) -> List[str]:
  """Convierte strings con formato de lista de Python a listas de forma segura."""
  if not isinstance(valor, str) or pd.isna(valor):
    return []
  try:
    res = ast.literal_eval(valor)
    return [item for item in res if isinstance(item, str)] if isinstance(res, list) else []
  except (ValueError, SyntaxError):
    return []


def sanear_jugadores_y_duracion(
    df: pd.DataFrame, max_duracion: int = 720
) -> pd.DataFrame:
  """Corrige jugadores y duración, devolviendo solo valores utilizables."""
  df_out = df.copy()
  columnas_numericas = [
      "minplayers",
      "maxplayers",
      "playingtime",
      "minplaytime",
      "maxplaytime",
  ]
  for columna in columnas_numericas:
    if columna in df_out:
      df_out[columna] = pd.to_numeric(df_out[columna], errors="coerce")
  df_out["playingtime"] = df_out["playingtime"].astype(float)

  # 1. Corrección de inversión minplayers > maxplayers
  filtro_invertido = df_out["minplayers"] > df_out["maxplayers"]
  if filtro_invertido.any():
    df_out.loc[filtro_invertido, ["minplayers", "maxplayers"]] = df_out.loc[
        filtro_invertido, ["maxplayers", "minplayers"]
    ].values

  # 2. Imputación de ceros en jugadores
  df_out["minplayers"] = df_out["minplayers"].replace(0, 1).fillna(1)
  df_out["maxplayers"] = np.where(
      df_out["maxplayers"].isna() | (df_out["maxplayers"] <= 0),
      df_out["minplayers"],
      df_out["maxplayers"],
  )
  df_out["maxplayers_atipico"] = df_out["maxplayers"] > MAX_JUGADORES_ANALISIS
  df_out["maxplayers"] = df_out["maxplayers"].clip(upper=MAX_JUGADORES_ANALISIS)

  # 3. Tratamiento de duración (playingtime)
  # Si playingtime == 0 pero min/max play time tienen dato positivo, imputar promedio
  mask_cero_playtime = df_out["playingtime"] <= 0
  estimado_playtime = (df_out["minplaytime"] + df_out["maxplaytime"]) / 2
  df_out.loc[mask_cero_playtime, "playingtime"] = np.where(
      estimado_playtime[mask_cero_playtime] > 0,
      estimado_playtime[mask_cero_playtime],
      np.nan,
  )

  # Imputar duraciones nulas restantes con la mediana global (robusta a outliers)
  mediana_duracion = df_out["playingtime"].median()
  if pd.isna(mediana_duracion):
    mediana_duracion = 1
  df_out["playingtime"] = df_out["playingtime"].fillna(mediana_duracion)

  # 4. Excluir outliers de duración extrema (> 720 minutos / 12 horas)
  df_out = df_out[df_out["playingtime"] <= max_duracion].copy()

  return df_out


def categorizar_y_sanear_anios(df: pd.DataFrame) -> pd.DataFrame:
  """Clasifica juegos históricos e imputa años cero cuando el registro es recuperable."""
  df_out = df.copy()

  # Marcar históricos (previos a 1900 o negativos)
  df_out["year_consolidated"] = pd.to_numeric(
      df_out["year_consolidated"], errors="coerce"
  )
  df_out["es_historico"] = df_out["year_consolidated"] < 1900

  # Si año es 0 o NaN y tiene información sustancial, imputar con mediana
  mask_anio_invalido = (df_out["year_consolidated"] <= 0) | (
      df_out["year_consolidated"].isna()
  )
  mediana_anio = df_out.loc[~mask_anio_invalido, "year_consolidated"].median()
  if not pd.isna(mediana_anio):
    df_out.loc[mask_anio_invalido, "year_consolidated"] = mediana_anio

  return df_out


def ejecutar_pipeline_limpieza(
    min_votos: int = 30, archivo_salida: str = "juegos_limpios.csv"
) -> pd.DataFrame:
  """Ejecuta el flujo completo de saneamiento y guarda el dataset procesado."""
  df_ratings, df_details = descargar_datos()
  df_merged = integrar_fuentes(df_ratings, df_details)

  # Filtro de representatividad estadística
  df_filtrado = df_merged[df_merged["users_rated"] >= min_votos].copy()

  # Saneamiento de variables numéricas y cronológicas
  df_saneado = sanear_jugadores_y_duracion(df_filtrado)
  df_final = categorizar_y_sanear_anios(df_saneado)

  # Parseo de listas
  columnas_lista = [
      "boardgamecategory",
      "boardgamemechanic",
      "boardgamedesigner",
      "boardgameartist",
      "boardgamepublisher",
  ]
  for col in columnas_lista:
    parsed_col = col + "_parsed"
    if col in df_final.columns:
      df_final[parsed_col] = df_final[col].apply(parsear_lista_segura)
    else:
      df_final[parsed_col] = [[] for _ in range(len(df_final))]

  # Variables de densidad de diseño
  df_final["num_mecanicas"] = df_final["boardgamemechanic_parsed"].apply(len)
  df_final["num_categorias"] = df_final["boardgamecategory_parsed"].apply(len)

  df_final.to_csv(archivo_salida, index=False)
  print(
      f"[OK] Dataset procesado guardado en: {archivo_salida} con"
      f" {df_final.shape[0]} filas y {df_final.shape[1]} columnas."
  )
  return df_final


if __name__ == "__main__":
  ejecutar_pipeline_limpieza()