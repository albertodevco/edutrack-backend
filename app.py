"""API Flask de EduTrack: sirve el modelo de presencia de casos foco (NEE).

Endpoints:
- ``GET  /``        información del servicio y del modelo.
- ``GET  /salud``   health check para Render.
- ``POST /predecir`` predicción con validación estricta (422 si es inválida).
"""

from __future__ import annotations

import json
import logging
import os

import sklearn
from flask import Flask, jsonify, request
from flask_cors import CORS
from pydantic import ValidationError

from model_service import informacion_modelo, predecir
from schemas import PrediccionRequest

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("edutrack.api")

app = Flask(__name__)
app.json.ensure_ascii = False
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

# Orígenes permitidos: "*" por defecto (demo), o lista separada por comas.
_origenes = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
CORS(app, resources={r"/*": {"origins": _origenes or "*"}})

# Carga y calentamiento del modelo al arrancar: si falta un artefacto, el
# despliegue falla de inmediato en lugar de fallar en la primera petición.
_INFORMACION = informacion_modelo()
logger.info(
    "Modelo listo: %s (%d features). sklearn=%s",
    _INFORMACION["modelo"],
    _INFORMACION["n_features"],
    sklearn.__version__,
)


@app.get("/")
@app.get("/salud")
def salud():
    return jsonify(
        {
            **_INFORMACION,
            "version_sklearn": sklearn.__version__,
            "endpoints": {"predecir": "POST /predecir"},
        }
    )


@app.post("/predecir")
def predecir_endpoint():
    cuerpo = request.get_json(silent=True)
    if not isinstance(cuerpo, dict):
        return (
            jsonify(
                {
                    "detail": [
                        {
                            "loc": ["body"],
                            "msg": "Se esperaba un objeto JSON válido.",
                            "type": "json_invalid",
                        }
                    ]
                }
            ),
            422,
        )

    try:
        datos = PrediccionRequest.model_validate(cuerpo)
    except ValidationError as exc:
        return jsonify({"detail": json.loads(exc.json(include_url=False))}), 422

    try:
        return jsonify(predecir(datos))
    except Exception:  # noqa: BLE001 - se registra y no se filtra el detalle
        logger.exception("Error ejecutando el modelo")
        return (
            jsonify({"detail": "Error interno ejecutando el modelo."}),
            500,
        )


@app.errorhandler(405)
def metodo_no_permitido(_error):
    return jsonify({"detail": "Método no permitido."}), 405


if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=puerto)
