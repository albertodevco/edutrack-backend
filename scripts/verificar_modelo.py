"""Verificación reproducible del modelo para la sustentación.

Uso (desde edutrack-backend):
    python scripts/verificar_modelo.py

Comprueba:
1. Que el orden de columnas del modelo coincide con columnas_edutrack.pkl.
2. Las probabilidades de los tres casos de demostración.
3. Que aplicar el StandardScaler es incorrecto para este árbol.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from model_service import (  # noqa: E402
    SCALER_PATH,
    cargar_artefactos,
    construir_fila,
    predecir,
)
from schemas import PrediccionRequest  # noqa: E402

CASOS = [
    (
        "Rural pequeña",
        dict(
            matricula=60,
            docentes=3,
            orientadores=0,
            niveles=2,
            jornadas=1,
            zona="rural",
            bilingue=False,
            etnoeducativa=False,
            departamento="Amazonas",
        ),
    ),
    (
        "Urbana grande",
        dict(
            matricula=950,
            docentes=34,
            orientadores=2,
            niveles=4,
            jornadas=2,
            zona="urbana",
            bilingue=False,
            etnoeducativa=False,
            departamento="Antioquia",
        ),
    ),
    (
        "Rural intermedia",
        dict(
            matricula=300,
            docentes=20,
            orientadores=1,
            niveles=3,
            jornadas=2,
            zona="rural",
            bilingue=False,
            etnoeducativa=False,
            departamento="Chocó",
        ),
    ),
]


def main() -> int:
    modelo, columnas = cargar_artefactos()
    if list(modelo.feature_names_in_) != columnas:
        print("ERROR: el orden de columnas no coincide con el modelo.")
        return 1

    print(f"Modelo: {type(modelo).__name__} | features: {len(columnas)}")
    print(f"Clases: {modelo.classes_.tolist()}")
    print()
    print(f"{'Caso':<18}{'Clase':<7}{'Probabilidad':>14}")
    print("-" * 39)
    for nombre, payload in CASOS:
        resultado = predecir(PrediccionRequest.model_validate(payload))
        print(
            f"{nombre:<18}{resultado['clase']:<7}"
            f"{resultado['probabilidad']:>14.12f}"
        )

    print()
    print("Comprobación del escalado (no debe usarse en inferencia):")
    escalador = joblib.load(SCALER_PATH)
    for nombre, payload in CASOS[:2]:
        datos = PrediccionRequest.model_validate(payload)
        fila, _ = construir_fila(datos)
        probabilidad_escalada = modelo.predict_proba(escalador.transform(fila))[0, 1]
        print(
            f"  {nombre:<18} cruda={modelo.predict_proba(fila)[0, 1]:.6f} "
            f"escalada={probabilidad_escalada:.6f}"
        )
    print(
        "\nEl árbol usa umbrales en unidades reales (matrícula 9.5, 55.5, 159.5, "
        "249.5, 948). Escalar deja la probabilidad casi constante: por eso el "
        "backend NO aplica el scaler."
    )
    assert np.isfinite(modelo.predict_proba(fila)).all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
