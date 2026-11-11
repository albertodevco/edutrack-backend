# EduTrack · Backend

API Flask que sirve el modelo `DecisionTreeClassifier` de EduTrack. Recibe los
parámetros estructurales de una sede, construye el vector de **41 columnas** en
el orden exacto del entrenamiento y devuelve la clase y la probabilidad.

Forma parte de un sistema de dos repositorios:

```text
edutrack-frontend (Supabase Auth + dashboard)  ──POST /predecir──►  edutrack-backend (Render)
            │                                                              │
            └────────────── tabla analisis_sede (RLS) ◄────────────────────┘
```

| Repositorio | Contenido |
|---|---|
| `edutrack-backend` (este) | API, modelo `.pkl`, pruebas, CI |
| `edutrack-frontend` | login, dashboard, `config.js` y SQL de Supabase |

## Requisitos

- Python 3.12 (ver `.python-version` / `runtime.txt`)
- Dependencias de `requirements.txt` (scikit-learn **1.6.1**, la versión con la
  que se serializó el modelo)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python app.py                  # http://localhost:5000
```

## Endpoints

### `GET /` · `GET /salud`

Metadatos del servicio y del modelo (features, clases, versión de sklearn).

### `POST /predecir`

```json
{
  "matricula": 950,
  "docentes": 34,
  "orientadores": 2,
  "niveles": 4,
  "jornadas": 2,
  "zona": "urbana",
  "bilingue": false,
  "etnoeducativa": false,
  "departamento": "Antioquia"
}
```

Respuesta `200`:

```json
{
  "clase": 1,
  "probabilidad": 0.7784891165172856,
  "etiqueta_clase": "Presencia estimada de casos foco (TEA, trastornos del aprendizaje o condiciones psicosociales) en la sede",
  "parametros_usados": {
    "matricula": 950,
    "docentes": 34,
    "orientadores": 2,
    "est_por_docente": 27.941176470588236,
    "niveles": 4,
    "jornadas": 2,
    "zona": "urbana",
    "zona_rural": 0,
    "bilingue": false,
    "etnoeducativa": false,
    "departamento": "Antioquia",
    "columna_departamento": "DEP_Antioquia"
  }
}
```

Errores de validación: `422` con `{"detail": [...]}` (formato Pydantic).
Error interno: `500` sin filtrar trazas al cliente.

## Reglas de validación

| Campo | Tipo | Rango |
|---|---|---|
| `matricula` | entero | 0 – 100 000 |
| `docentes` | entero | 1 – 5 000 (no puede ser 0) |
| `orientadores` | entero | 0 – 500 |
| `niveles` | entero | 1 – 6 |
| `jornadas` | entero | 1 – 5 |
| `zona` | literal | `"rural"` o `"urbana"` (minúsculas) |
| `bilingue`, `etnoeducativa` | booleano | `true` / `false` |
| `departamento` | enum | 33 valores exactos; `Amazonas` es la categoría base |

Los campos extra se rechazan (`extra="forbid"`). No hay valores por defecto
silenciosos.

## Decisiones sobre el modelo (importante)

1. **El scaler NO se aplica en inferencia.** Los umbrales del árbol de
   `MATRICULA` son 9.5, 21.5, 55.5, 159.5, 249.5 y 948, es decir, unidades
   reales. Aplicar `StandardScaler` deja la probabilidad plana en `0.3125` para
   casi cualquier sede. `scaler_edutrack.pkl` se conserva solo como referencia.
2. **Amazonas es la categoría base**: no tiene dummy y todas las `DEP_*` quedan
   en 0.
3. **El orden de las 41 columnas** se toma siempre de `columnas_edutrack.pkl`.
4. **`EST_POR_DOCENTE`** se calcula en el backend como `MATRICULA / N_DOCENTES`.
5. **Clase 1**: el modelo estima la presencia de casos foco. La evidencia
   disponible (la probabilidad crece con la matrícula y con la presencia de
   orientadores) apunta a “la sede registra al menos un caso foco”, pero debe
   confirmarse contra el notebook de entrenamiento antes de presentarlo como
   diagnóstico.

## Pruebas

```bash
pip install -r requirements-dev.txt
pytest -q
```

Las pruebas comparan la API contra el modelo local con tolerancia `1e-9`,
verifican los 33 departamentos, los rangos y que el scaler no se aplique.
También hay una verificación reproducible para la sustentación:

```bash
python scripts/verificar_modelo.py
```

## Variables de entorno

| Variable | Default | Uso |
|---|---|---|
| `PORT` | `5000` | Puerto de Flask/gunicorn |
| `ALLOWED_ORIGINS` | `*` | Orígenes CORS separados por comas |
| `LOG_LEVEL` | `INFO` | Nivel de logging |

## Despliegue en Render

- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app`
- Python: `3.12.7` (`.python-version` / `runtime.txt`)
- Mantener `scikit-learn==1.6.1` para evitar incompatibilidades al cargar el
  `.pkl`.

Verificación rápida:

```bash
curl -s https://TU-SERVICIO.onrender.com/salud

curl -s -X POST https://TU-SERVICIO.onrender.com/predecir \
  -H "Content-Type: application/json" \
  -d '{"matricula":950,"docentes":34,"orientadores":2,"niveles":4,"jornadas":2,"zona":"urbana","bilingue":false,"etnoeducativa":false,"departamento":"Antioquia"}'
```

> El frontend nuevo envía `bilingue`/`etnoeducativa` como booleanos y
> `departamento`. **Despliega este backend antes de publicar el frontend**; la
> versión anterior de la API no conoce esos campos.

## Subir a GitHub

El repositorio ya está inicializado y con los cambios preparados. Solo falta
hacer commit y push:

```powershell
cd "C:\Users\Admin\Documents\Inteligencia Artificial\edutrack-backend"
git add -A
git commit -m "feat: validación Pydantic, vector de 41 columnas y modelo sin escalado"
git push origin main
```

Guía completa (autenticación, Render, errores frecuentes y seguridad):
**[docs/PUBLICAR_EN_GITHUB.md](docs/PUBLICAR_EN_GITHUB.md)**.

Cada push a `main` ejecuta las 50 pruebas con GitHub Actions
(`.github/workflows/tests.yml`).

## Documentación

- `docs/DIAGNOSTICO.md` — comparación del código anterior contra los parámetros
  reales del modelo y la corrección del escalado.
- `docs/PUBLICAR_EN_GITHUB.md` — publicación y despliegue paso a paso.
