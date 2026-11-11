"""Pruebas de contrato y de regresión del modelo.

Ejecutar desde edutrack-backend:
    pytest -q
"""

from __future__ import annotations

import pytest

from app import app
from model_service import departamentos, predecir
from schemas import PrediccionRequest

# Valores de referencia calculados directamente desde los .pkl del repositorio
# con scikit-learn 1.6.1 (sin escalar). Tolerancia exigida: 1e-9.
CASOS = [
    pytest.param(
        {
            "matricula": 60,
            "docentes": 3,
            "orientadores": 0,
            "niveles": 2,
            "jornadas": 1,
            "zona": "rural",
            "bilingue": False,
            "etnoeducativa": False,
            "departamento": "Amazonas",
        },
        0.11713286713286714,
        0,
        id="rural-pequena",
    ),
    pytest.param(
        {
            "matricula": 950,
            "docentes": 34,
            "orientadores": 2,
            "niveles": 4,
            "jornadas": 2,
            "zona": "urbana",
            "bilingue": False,
            "etnoeducativa": False,
            "departamento": "Antioquia",
        },
        0.7784891165172856,
        1,
        id="urbana-grande",
    ),
    pytest.param(
        {
            "matricula": 300,
            "docentes": 20,
            "orientadores": 1,
            "niveles": 3,
            "jornadas": 2,
            "zona": "rural",
            "bilingue": False,
            "etnoeducativa": False,
            "departamento": "Chocó",
        },
        0.43349753694581283,
        0,
        id="rural-intermedia",
    ),
]


@pytest.fixture(scope="module")
def cliente():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


@pytest.mark.parametrize(("payload", "esperado", "clase"), CASOS)
def test_api_coincide_con_modelo_local(cliente, payload, esperado, clase):
    respuesta = cliente.post("/predecir", json=payload)

    assert respuesta.status_code == 200, respuesta.get_json()
    datos = respuesta.get_json()

    local = predecir(PrediccionRequest.model_validate(payload))

    assert datos["probabilidad"] == pytest.approx(esperado, abs=1e-9)
    assert datos["probabilidad"] == pytest.approx(local["probabilidad"], abs=1e-9)
    assert datos["clase"] == clase
    assert datos["parametros_usados"]["departamento"] == payload["departamento"]
    assert datos["parametros_usados"]["matricula"] == payload["matricula"]


def test_est_por_docente_se_calcula_en_backend(cliente):
    payload = dict(CASOS[1].values[0])
    datos = cliente.post("/predecir", json=payload).get_json()
    assert datos["parametros_usados"]["est_por_docente"] == pytest.approx(
        payload["matricula"] / payload["docentes"], abs=1e-9
    )


def test_amazonas_es_categoria_base(cliente):
    payload = dict(CASOS[0].values[0])
    datos = cliente.post("/predecir", json=payload).get_json()
    assert "categoría base" in datos["parametros_usados"]["columna_departamento"]


@pytest.mark.parametrize("departamento", departamentos())
def test_los_33_departamentos_son_aceptados(cliente, departamento):
    payload = dict(CASOS[0].values[0])
    payload["departamento"] = departamento
    respuesta = cliente.post("/predecir", json=payload)
    assert respuesta.status_code == 200, (departamento, respuesta.get_json())


@pytest.mark.parametrize(
    ("cambio", "motivo"),
    [
        ({"docentes": 0}, "docentes no puede ser 0"),
        ({"matricula": -1}, "matrícula negativa"),
        ({"niveles": 0}, "niveles fuera de rango"),
        ({"jornadas": 9}, "jornadas fuera de rango"),
        ({"zona": "Rural"}, "zona no normalizada"),
        ({"departamento": "Atlántida"}, "departamento inexistente"),
        ({"bilingue": "tal vez"}, "booleano inválido"),
        ({"campo_extra": 1}, "campo extra"),
    ],
)
def test_validacion_estricta_devuelve_422(cliente, cambio, motivo):
    payload = dict(CASOS[0].values[0])
    payload.update(cambio)
    respuesta = cliente.post("/predecir", json=payload)
    assert respuesta.status_code == 422, motivo
    assert "detail" in respuesta.get_json()


def test_campo_faltante_devuelve_422(cliente):
    payload = dict(CASOS[0].values[0])
    del payload["matricula"]
    assert cliente.post("/predecir", json=payload).status_code == 422


def test_cuerpo_no_json_devuelve_422(cliente):
    respuesta = cliente.post(
        "/predecir", data="no es json", content_type="text/plain"
    )
    assert respuesta.status_code == 422


def test_salud_expone_metadatos(cliente):
    datos = cliente.get("/salud").get_json()
    assert datos["estado"] == "ok"
    assert datos["n_features"] == 41
    assert datos["clases"] == [0, 1]
    assert datos["escalado_en_inferencia"] is False


def test_el_scaler_no_se_aplica(cliente):
    """Si se aplicara el scaler, la probabilidad quedaría plana en 0.3125."""
    payload = dict(CASOS[0].values[0])
    datos = cliente.post("/predecir", json=payload).get_json()
    assert datos["probabilidad"] != pytest.approx(0.3125, abs=1e-6)
