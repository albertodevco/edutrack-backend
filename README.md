# EduTrack · Backend

API Flask que sirve el modelo `DecisionTreeClassifier` de EduTrack. Recibe los
parámetros estructurales de una sede, construye el vector de **41 columnas** en
el orden exacto del entrenamiento y devuelve la clase y la probabilidad.

Forma parte de un sistema de dos repositorios:



| Repositorio | Contenido |
|---|---|
| `edutrack-backend` (este) | API, modelo `.pkl`, pruebas, CI |
| `edutrack-frontend` | login, dashboard, `config.js` y SQL de Supabase |

## Requisitos

- Python 3.12 (ver `.python-version` / `runtime.txt`)
- Dependencias de `requirements.txt` (scikit-learn **1.6.1**, la versión con la
  que se serializó el modelo)

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

## Decisiones sobre el modelo

1. **El scaler NO se aplica en inferencia.** El árbol se entrenó con valores en
   unidades reales (matrícula en estudiantes, sin estandarizar). Aplicar
   `StandardScaler` produce probabilidades incoherentes: una sede urbana de 950
   estudiantes daría 13.9 % en vez de 96.7 %. `scaler_edutrack.pkl` se conserva
   solo como referencia.
2. **Amazonas es la categoría base**: no tiene dummy y todas las `DEP_*` quedan
   en 0.
3. **El orden de las 41 columnas** se toma siempre de `columnas_edutrack.pkl`.
4. **`EST_POR_DOCENTE`** se calcula en el backend como `MATRICULA / N_DOCENTES`.
5. **Clase 1**: la sede reporta al menos un caso foco
   (`REPORTA_FOCO = N_FOCO > 0`), confirmado en el notebook de entrenamiento.
6. **Resolución.** El modelo original (`max_depth=6`, `min_samples_leaf=30`)
   tenía 51 hojas y solo 50 probabilidades distintas, así que muchas sedes
   obtenían el mismo valor. Se reentrenó con `max_depth=24` y
   `min_samples_leaf=40`: **515 hojas**, **339 probabilidades distintas** y
   **100 umbrales de matrícula**. Métricas de prueba: exactitud `0.8710`,
   F1 `0.5943`, AUC `0.8745` (antes `0.8674` / `0.5160` / `0.8736`).



## Pruebas



Las pruebas comparan la API contra el modelo local con tolerancia `1e-9`,
verifican los 33 departamentos, los rangos y que el scaler no se aplique.
También hay una verificación reproducible para la sustentación:


## Variables de entorno

| Variable | Default | Uso |
|---|---|---|
| `PORT` | `5000` | Puerto de Flask/gunicorn |
| `ALLOWED_ORIGINS` | `*` | Orígenes CORS separados por comas |
| `LOG_LEVEL` | `INFO` | Nivel de logging |

