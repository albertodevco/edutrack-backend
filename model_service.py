"""Carga única de artefactos y construcción del vector de entrada del modelo.

Decisiones importantes (verificadas contra los .pkl del repositorio):

1. El ``DecisionTreeClassifier`` fue entrenado con valores en unidades reales
   (matrícula en estudiantes, sin estandarizar). Sus umbrales solo tienen
   sentido sin escalar: aplicar ``scaler_edutrack.pkl`` produce probabilidades
   incoherentes (una sede de 950 estudiantes daría 13.9 %). Por eso el scaler
   NO se usa en inferencia; se conserva como referencia del entrenamiento.
2. El orden de las 41 columnas sale siempre de ``columnas_edutrack.pkl``.
3. Amazonas es la categoría base: no tiene dummy y todas las DEP_* quedan en 0.
4. El modelo se reentrenó con ``max_depth=24`` y ``min_samples_leaf=40``
   (515 hojas y 339 probabilidades distintas, frente a 51 hojas y 50 valores
   del modelo original) para que la respuesta no se repita entre sedes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_edutrack.pkl"
COLUMNS_PATH = BASE_DIR / "columnas_edutrack.pkl"
SCALER_PATH = BASE_DIR / "scaler_edutrack.pkl"  # referencia, no se usa

#: Significado de la clase 1, confirmado en el notebook de entrenamiento:
#: ``REPORTA_FOCO = (N_FOCO > 0)``, es decir, la sede reporta al menos un caso
#: foco (TEA, trastorno del aprendizaje o condición psicosocial).
SIGNIFICADO_CLASE_1 = (
    "Presencia estimada de casos foco (TEA, trastornos del aprendizaje o "
    "condiciones psicosociales) en la sede"
)

_modelo: Any | None = None
_columnas: list[str] | None = None
_departamentos: list[str] | None = None


class ModeloNoDisponible(RuntimeError):
    """Los artefactos del modelo no pudieron cargarse."""


def cargar_artefactos() -> tuple[Any, list[str]]:
    """Carga el modelo una sola vez y devuelve (modelo, columnas)."""
    global _modelo, _columnas, _departamentos

    if _modelo is None or _columnas is None:
        if not MODEL_PATH.exists() or not COLUMNS_PATH.exists():
            raise ModeloNoDisponible(
                f"Faltan artefactos del modelo en {BASE_DIR}. "
                "Se requieren modelo_edutrack.pkl y columnas_edutrack.pkl."
            )
        inicio = time.perf_counter()
        _modelo = joblib.load(MODEL_PATH)
        _columnas = [str(c) for c in joblib.load(COLUMNS_PATH)]
        _departamentos = ["Amazonas"] + [
            c[len("DEP_") :] for c in _columnas if c.startswith("DEP_")
        ]
        logger.info(
            "Modelo cargado en %.0f ms (%s, %d columnas)",
            (time.perf_counter() - inicio) * 1000,
            type(_modelo).__name__,
            len(_columnas),
        )
    return _modelo, _columnas


def departamentos() -> list[str]:
    """Lista de los 33 departamentos, en el orden de las columnas del modelo."""
    if _departamentos is None:
        cargar_artefactos()
    assert _departamentos is not None
    return list(_departamentos)


def construir_fila(datos: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Convierte los 9 parámetros capturados en el vector de 41 columnas.

    ``datos`` es un ``PrediccionRequest`` (se accede por atributos para no
    acoplar este módulo al esquema HTTP).
    """
    modelo, columnas = cargar_artefactos()

    est_por_docente = datos.matricula / datos.docentes
    es_rural = datos.zona == "rural"

    fila = dict.fromkeys(columnas, 0.0)
    fila["MATRICULA"] = float(datos.matricula)
    fila["N_DOCENTES"] = float(datos.docentes)
    fila["N_ORIENTADORES"] = float(datos.orientadores)
    fila["EST_POR_DOCENTE"] = float(est_por_docente)
    fila["N_NIVELES"] = float(datos.niveles)
    fila["N_JORNADAS"] = float(datos.jornadas)
    fila["ZONA_RURAL"] = 1.0 if es_rural else 0.0
    fila["BILINGUE_SI"] = 1.0 if datos.bilingue else 0.0
    fila["ETNOEDUC_SI"] = 1.0 if datos.etnoeducativa else 0.0

    nombre_departamento = datos.departamento.value
    columna_departamento: str | None = None
    if nombre_departamento != "Amazonas":
        columna_departamento = f"DEP_{nombre_departamento}"
        if columna_departamento not in fila:
            raise ValueError(
                f"El departamento {nombre_departamento!r} no tiene columna en el "
                f"modelo ({columna_departamento!r})."
            )
        fila[columna_departamento] = 1.0

    vector = np.array([[fila[c] for c in columnas]], dtype=float)

    if vector.shape[1] != getattr(modelo, "n_features_in_", len(columnas)):
        raise ValueError(
            "El vector construido no coincide con las features del modelo: "
            f"{vector.shape[1]} vs {getattr(modelo, 'n_features_in_', '?')}."
        )

    parametros_usados = {
        "matricula": datos.matricula,
        "docentes": datos.docentes,
        "orientadores": datos.orientadores,
        "est_por_docente": est_por_docente,
        "niveles": datos.niveles,
        "jornadas": datos.jornadas,
        "zona": datos.zona,
        "zona_rural": 1 if es_rural else 0,
        "bilingue": datos.bilingue,
        "etnoeducativa": datos.etnoeducativa,
        "departamento": nombre_departamento,
        "columna_departamento": columna_departamento
        or "Amazonas (categoría base: sin dummy)",
    }

    return vector, parametros_usados


def predecir(datos: Any) -> dict[str, Any]:
    """Ejecuta el modelo y devuelve clase, probabilidad y parámetros usados."""
    modelo, _ = cargar_artefactos()
    vector, parametros_usados = construir_fila(datos)

    probabilidades = modelo.predict_proba(vector)[0]
    clases = [int(c) for c in modelo.classes_]
    if 1 not in clases:
        raise ValueError("El modelo no tiene la clase 1 entre sus clases.")

    indice_clase_1 = clases.index(1)
    probabilidad = float(probabilidades[indice_clase_1])
    clase = int(modelo.predict(vector)[0])

    return {
        "clase": clase,
        "probabilidad": probabilidad,
        "etiqueta_clase": SIGNIFICADO_CLASE_1,
        "parametros_usados": parametros_usados,
    }


def informacion_modelo() -> dict[str, Any]:
    """Metadatos del modelo para el endpoint de salud."""
    modelo, columnas = cargar_artefactos()
    return {
        "estado": "ok",
        "modelo": type(modelo).__name__,
        "n_features": len(columnas),
        "clases": [int(c) for c in modelo.classes_],
        "hojas": int(modelo.get_n_leaves()),
        "profundidad": int(modelo.get_depth()),
        "significado_clase_1": SIGNIFICADO_CLASE_1,
        "escalado_en_inferencia": False,
        "nota_escalado": (
            "El árbol se entrenó con valores en unidades reales; el scaler no "
            "se aplica en inferencia (ver model_service.py)."
        ),
    }
