"""Reentrena el arbol de EduTrack con mas resolucion.

Reproduce el pipeline del notebook ``EduTrack_Proyecto 1.2.ipynb`` (celdas 2 a
14) y permite explorar ``max_depth`` / ``min_samples_leaf`` antes de exportar
los artefactos que usa la API.

El objetivo es que la probabilidad deje de repetirse: el arbol original
(``max_depth=6``, ``min_samples_leaf=30``) solo tenia 51 hojas y devolvia 50
valores distintos para cualquier sede.

Uso
---
Explorar la rejilla (no escribe nada en el backend)::

    python scripts/reentrenar_modelo.py --zip "C:/ruta/BDATOS-EDUC-2024.zip"

Exportar una configuracion concreta::

    python scripts/reentrenar_modelo.py --zip "..." --exportar \
        --max-depth 12 --min-samples-leaf 5

El script debe ejecutarse con las mismas versiones que Render
(``scikit-learn==1.6.1``) para que el ``.pkl`` sea compatible.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FOCO = ["TEA", "Aprendizaje/Comportamiento", "Psicosocial"]

COLUMNAS_NUM = [
    "MATRICULA",
    "N_DOCENTES",
    "N_ORIENTADORES",
    "EST_POR_DOCENTE",
    "N_NIVELES",
    "N_JORNADAS",
    "ZONA_RURAL",
    "BILINGUE_SI",
    "ETNOEDUC_SI",
]

#: Categoria que ``get_dummies(drop_first=True)`` deja como base.
DEPARTAMENTO_BASE = "Amazonas"

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Construccion del dataset (celdas 2 a 14 del notebook)
# ---------------------------------------------------------------------------
def tipo_corto(nombre: str) -> str:
    n = nombre.lower()
    if "autismo" in n:
        return "TEA"
    if "trastornos espec" in n:
        return "Aprendizaje/Comportamiento"
    if "psicosocial" in n:
        return "Psicosocial"
    if "cognitiva" in n:
        return "Cognitiva/Intelectual"
    if "ltiple" in n:
        return "Múltiple"
    if "sica" in n:
        return "Física"
    if "visual" in n:
        return "Visual"
    return "Auditiva/Sordera"


def categoria_corta(nombre: str) -> str:
    n = nombre.lower()
    if n.startswith("docentes de aula"):
        return "N_DOCENTES"
    if n.startswith("docentes orientadores"):
        return "N_ORIENTADORES"
    return "OTRO"


def _buscar_csv(zf: zipfile.ZipFile, palabra: str) -> str:
    """Devuelve el nombre interno del primer CSV que contiene ``palabra``."""
    for nombre in zf.namelist():
        n = nombre.lower()
        if n.endswith(".csv") and palabra in n:
            return nombre
    raise FileNotFoundError(f"No hay ningun CSV con {palabra!r} dentro del zip.")


def construir_dataset(ruta_zip: Path) -> pd.DataFrame:
    with zipfile.ZipFile(ruta_zip) as zf:
        disc = pd.read_csv(
            zf.open(_buscar_csv(zf, "discapacidad")),
            encoding="utf-8-sig",
            dtype={"SEDE_CODIGO": str},
            usecols=[
                "SEDE_CODIGO",
                "LIMITACION_NOMBRE",
                "JORNLIM_CANTIDAD_HOMBRE",
                "JORNLIM_CANTIDAD_MUJER",
            ],
        )
        matr = pd.read_csv(
            zf.open(_buscar_csv(zf, "matriculados por jornadas")),
            encoding="utf-8-sig",
            dtype={"SEDE_CODIGO": str},
            usecols=["SEDE_CODIGO", "SEDEALUM_CANTIDAD"],
        )
        cara = pd.read_csv(
            zf.open(_buscar_csv(zf, "tula")),
            encoding="utf-8-sig",
            dtype={"SEDE_CODIGO": str},
            usecols=[
                "SEDE_CODIGO",
                "NOVEDAD_NOMBRE",
                "SECTOR_NOMBRE",
                "DEPTO",
                "AREA_NOMBRE",
                "ES_BILINGUE",
                "ES_ETNOEDUCATIVA",
                "SEDE_TOTAL_OFRE_ANTE_NIVEL",
                "SEDE_TOTAL_OFRE_ANTE_JORN",
            ],
        )
        pers = pd.read_csv(
            zf.open(_buscar_csv(zf, "personal ocupado")),
            encoding="utf-8-sig",
            dtype={"SEDE_CODIGO": str},
            usecols=[
                "SEDE_CODIGO",
                "CATEGORIA_NOMBRE",
                "SEDEPERO_CANTIDAD_HOMBRE",
                "SEDEPERO_CANTIDAD_MUJER",
            ],
        )

    # --- condiciones foco por sede (celdas 4 y 8) ---
    disc["TOTAL"] = (
        disc["JORNLIM_CANTIDAD_HOMBRE"] + disc["JORNLIM_CANTIDAD_MUJER"]
    )
    disc["TIPO"] = disc["LIMITACION_NOMBRE"].map(tipo_corto)

    cond_sede = disc.pivot_table(
        index="SEDE_CODIGO",
        columns="TIPO",
        values="TOTAL",
        aggfunc="sum",
        fill_value=0,
    )
    cond_sede["N_FOCO"] = cond_sede[FOCO].sum(axis=1)
    cond_sede["N_COND_TOTAL"] = cond_sede.drop(columns="N_FOCO").sum(axis=1)
    cond_sede = cond_sede[["N_FOCO", "N_COND_TOTAL"]].reset_index()

    # --- sedes oficiales que rinden (celda 7) ---
    sedes = cara[
        (cara["NOVEDAD_NOMBRE"] == "Rinde") & (cara["SECTOR_NOMBRE"] == "Oficial")
    ][
        [
            "SEDE_CODIGO",
            "DEPTO",
            "AREA_NOMBRE",
            "ES_BILINGUE",
            "ES_ETNOEDUCATIVA",
            "SEDE_TOTAL_OFRE_ANTE_NIVEL",
            "SEDE_TOTAL_OFRE_ANTE_JORN",
        ]
    ].copy()
    sedes.columns = [
        "SEDE_CODIGO",
        "DEPTO",
        "ZONA",
        "BILINGUE",
        "ETNOEDUC",
        "N_NIVELES",
        "N_JORNADAS",
    ]

    # --- matricula y personal por sede (celda 8) ---
    mat_sede = (
        matr.groupby("SEDE_CODIGO")["SEDEALUM_CANTIDAD"]
        .sum()
        .reset_index()
        .rename(columns={"SEDEALUM_CANTIDAD": "MATRICULA"})
    )

    pers["TOTAL"] = (
        pers["SEDEPERO_CANTIDAD_HOMBRE"] + pers["SEDEPERO_CANTIDAD_MUJER"]
    )
    pers["CAT"] = pers["CATEGORIA_NOMBRE"].map(categoria_corta)
    pers = pers[pers["CAT"] != "OTRO"]
    pers_sede = pers.pivot_table(
        index="SEDE_CODIGO",
        columns="CAT",
        values="TOTAL",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    df = sedes.merge(mat_sede, on="SEDE_CODIGO", how="left")
    df = df.merge(cond_sede, on="SEDE_CODIGO", how="left")
    df = df.merge(pers_sede, on="SEDE_CODIGO", how="left")

    for columna in ("N_FOCO", "N_COND_TOTAL", "N_DOCENTES", "N_ORIENTADORES"):
        df[columna] = df[columna].fillna(0)

    df = df[df["MATRICULA"] > 0].reset_index(drop=True)

    # --- variables derivadas y objetivo (celda 13) ---
    df["ZONA_RURAL"] = (df["ZONA"] == "Rural").astype(int)
    df["BILINGUE_SI"] = (df["BILINGUE"] == "SÍ").astype(int)
    df["ETNOEDUC_SI"] = (df["ETNOEDUC"] == "SÍ").astype(int)
    df["EST_POR_DOCENTE"] = df["MATRICULA"] / df["N_DOCENTES"].replace(0, np.nan)
    df["EST_POR_DOCENTE"] = df["EST_POR_DOCENTE"].fillna(df["MATRICULA"])
    df["REPORTA_FOCO"] = (df["N_FOCO"] > 0).astype(int)
    return df


def matriz_entrada(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    deptos = pd.get_dummies(df["DEPTO"], prefix="DEP", drop_first=True).astype(int)
    X = pd.concat([df[COLUMNAS_NUM], deptos], axis=1)
    y = df["REPORTA_FOCO"]
    return X, y


# ---------------------------------------------------------------------------
# Entrenamiento y metricas
# ---------------------------------------------------------------------------
def metricas(modelo: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = modelo.predict_proba(X_test)[:, 1]
    pred = modelo.predict(X_test)
    return {
        "exactitud": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, proba)),
        "hojas": int(modelo.get_n_leaves()),
        "profundidad": int(modelo.get_depth()),
        "probabilidades_distintas": int(len(np.unique(np.round(proba, 12)))),
        "probabilidad_min": float(proba.min()),
        "probabilidad_max": float(proba.max()),
    }


def explorar(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    profundidades: list[int | None],
    hojas_minimas: list[int],
) -> list[dict]:
    filas = []
    for profundidad in profundidades:
        for hoja in hojas_minimas:
            modelo = DecisionTreeClassifier(
                max_depth=profundidad,
                min_samples_leaf=hoja,
                random_state=42,
            )
            modelo.fit(X_train, y_train)
            fila = {"max_depth": profundidad, "min_samples_leaf": hoja}
            fila.update(metricas(modelo, X_test, y_test))
            filas.append(fila)
            print(
                f"  depth={str(profundidad):>4} leaf={hoja:>2}  "
                f"acc={fila['exactitud']:.4f}  f1={fila['f1']:.4f}  "
                f"auc={fila['auc']:.4f}  hojas={fila['hojas']:>5}  "
                f"probs={fila['probabilidades_distintas']:>4}",
                flush=True,
            )
    return filas


def sensibilidad(modelo: DecisionTreeClassifier, columnas: list[str]) -> dict:
    """Cuantos valores distintos devuelve al mover matricula y docentes."""
    base = {
        "matricula": 400,
        "docentes": 16,
        "orientadores": 0,
        "niveles": 6,
        "jornadas": 2,
        "zona": "urbana",
        "bilingue": False,
        "etnoeducativa": False,
        "departamento": "Atlántico",
    }

    def predecir(**cambios) -> float:
        datos = {**base, **cambios}
        fila = dict.fromkeys(columnas, 0.0)
        fila["MATRICULA"] = float(datos["matricula"])
        fila["N_DOCENTES"] = float(datos["docentes"])
        fila["N_ORIENTADORES"] = float(datos["orientadores"])
        fila["EST_POR_DOCENTE"] = datos["matricula"] / datos["docentes"]
        fila["N_NIVELES"] = float(datos["niveles"])
        fila["N_JORNADAS"] = float(datos["jornadas"])
        fila["ZONA_RURAL"] = 1.0 if datos["zona"] == "rural" else 0.0
        fila["BILINGUE_SI"] = 1.0 if datos["bilingue"] else 0.0
        fila["ETNOEDUC_SI"] = 1.0 if datos["etnoeducativa"] else 0.0
        departamento = datos["departamento"]
        if departamento != DEPARTAMENTO_BASE:
            fila["DEP_" + departamento] = 1.0
        vector = np.array([[fila[c] for c in columnas]], dtype=float)
        return float(modelo.predict_proba(vector)[0, 1])

    matriculas = [20, 40, 60, 80, 100, 120, 150, 180, 200, 220, 250, 280, 300,
                  350, 400, 450, 500, 550, 600, 650, 700, 800, 900, 1000, 1200]
    docentes = list(range(1, 41))
    probs_matricula = [round(predecir(matricula=v), 6) for v in matriculas]
    probs_docentes = [round(predecir(docentes=v), 6) for v in docentes]
    return {
        "base": round(predecir(), 6),
        "matriculas_probadas": matriculas,
        "probabilidades_por_matricula": probs_matricula,
        "matriculas_con_valor_distinto": len(set(probs_matricula)),
        "docentes_probados": docentes,
        "probabilidades_por_docentes": probs_docentes,
        "docentes_con_valor_distinto": len(set(probs_docentes)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True, help="Ruta del zip del DANE")
    parser.add_argument(
        "--exportar",
        action="store_true",
        help="Escribe modelo_edutrack.pkl y columnas_edutrack.pkl en el backend",
    )
    parser.add_argument("--max-depth", type=str, default="12")
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--out", type=Path, default=BASE_DIR)
    parser.add_argument(
        "--solo-exportar",
        action="store_true",
        help="Omite la rejilla y entrena directamente la configuracion indicada",
    )
    args = parser.parse_args()

    print(f"Leyendo {args.zip} ...", flush=True)
    df = construir_dataset(args.zip)
    X, y = matriz_entrada(df)

    print(f"Sedes: {len(df)} | features: {X.shape[1]} | clase 1: {y.mean():.4f}")
    if len(df) != 42299 or X.shape[1] != 41:
        print(
            "AVISO: el notebook original daba 42299 sedes y 41 features. "
            "Revisa que el zip sea el mismo antes de exportar.",
            flush=True,
        )

    columnas_antiguas = BASE_DIR / "columnas_edutrack.pkl"
    if columnas_antiguas.exists():
        anteriores = [str(c) for c in joblib.load(columnas_antiguas)]
        print(
            "Columnas identicas al modelo anterior:",
            anteriores == list(X.columns),
            flush=True,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Entrenamiento: {X_train.shape} | Prueba: {X_test.shape}", flush=True)

    if not args.solo_exportar:
        print("\nRejilla de hiperparametros:", flush=True)
        resultados = explorar(
            X_train, X_test, y_train, y_test,
            [6, 8, 10, 12, 14, 16, 20, None],
            [1, 2, 3, 5, 10, 20, 30],
        )
        csv = args.out / "docs" / "reentrenamiento_grid.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(resultados).to_csv(csv, index=False)
        print(f"\nRejilla guardada en {csv}", flush=True)

    if not args.exportar:
        print("\nExploracion terminada (sin --exportar no se toca el backend).")
        return 0

    profundidad = None if args.max_depth.lower() in {"none", "null"} else int(args.max_depth)
    print(
        f"\nEntrenando configuracion final: max_depth={profundidad}, "
        f"min_samples_leaf={args.min_samples_leaf}",
        flush=True,
    )
    modelo = DecisionTreeClassifier(
        max_depth=profundidad,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42,
    )
    modelo.fit(X_train, y_train)
    resumen = metricas(modelo, X_test, y_test)
    print(json.dumps(resumen, indent=2, ensure_ascii=False), flush=True)

    scaler = StandardScaler().fit(X_train)  # solo referencia, no se usa en inferencia

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, args.out / "modelo_edutrack.pkl")
    joblib.dump(list(X.columns), args.out / "columnas_edutrack.pkl")
    joblib.dump(scaler, args.out / "scaler_edutrack.pkl")

    detalle = {
        "hiperparametros": {
            "max_depth": profundidad,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": 42,
            "criterion": "gini",
        },
        "dataset": {
            "sedes": int(len(df)),
            "features": int(X.shape[1]),
            "entrenamiento": int(len(X_train)),
            "prueba": int(len(X_test)),
            "proporcion_clase_1": float(y.mean()),
        },
        "metricas_prueba": resumen,
        "importancias": sorted(
            (
                {"variable": columna, "importancia": float(importancia)}
                for columna, importancia in zip(X.columns, modelo.feature_importances_)
                if importancia > 0
            ),
            key=lambda item: item["importancia"],
            reverse=True,
        ),
        "sensibilidad": sensibilidad(modelo, list(X.columns)),
    }
    destino = args.out / "docs" / "metricas_modelo.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(detalle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nArtefactos escritos en {args.out}")
    print(f"Metricas en {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
