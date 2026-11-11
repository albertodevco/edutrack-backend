# Diagnóstico: código anterior vs. parámetros reales del modelo

Comparación solicitada en la hoja **Diagnóstico** de
`edutrack_parametros_modelo.xlsx`, ya corregida en el código actual.

## Estado final

| Parámetro | ¿Está en el formulario? | ¿Llega al payload? | ¿Llega al modelo? | Observación |
|---|---|---|---|---|
| `MATRICULA` | Sí | Sí | Sí | Entero 0 – 100 000 |
| `ZONA_RURAL` | Sí (select Rural/Urbana) | Sí (`zona`) | Sí (1/0) | Segunda variable en importancia (9,8 %) |
| `N_DOCENTES` | Sí | Sí | Sí | Entero ≥ 1; divisor de `EST_POR_DOCENTE` |
| `N_ORIENTADORES` | Sí | Sí | Sí | Entero ≥ 0 |
| `ETNOEDUC_SI` | Sí (switch) | Sí (booleano) | Sí (1/0) | Peso 0,99 % |
| `N_NIVELES` | Sí | Sí | Sí | Entero 1 – 6; peso 0,54 % |
| `N_JORNADAS` | Sí | Sí | Sí | Entero 1 – 5; peso 0,14 % |
| `BILINGUE_SI` | Sí (switch) | Sí (booleano) | Sí (1/0) | Peso 0,01 % |
| `EST_POR_DOCENTE` | No (calculado) | — | Sí | `MATRICULA / N_DOCENTES` en el backend |
| `departamento` (32 dummies) | Sí (select de 33) | Sí (`departamento`) | Sí | Se activa `DEP_<nombre>`; Amazonas deja todas en 0 |

Parámetros con algún “No”: **0**.

## Brechas encontradas en la versión anterior

1. **`casos_foco` / `casos_total` se capturaban y nunca se usaban.** No son
   columnas del modelo y el JS leía un inexistente `casos_reportados`, por lo
   que los casos siempre valían 0.
2. **`municipio` era obligatorio y no existía en el modelo.** El modelo usa
   solo departamento y zona.
3. **`departamento` era texto libre y no se enviaba.** Las 32 dummies quedaban
   siempre en 0 (todas las sedes se inferían como Amazonas).
4. **Los booleanos se enviaban como `"Sí"`/`"No"`** y el backend comparaba
   contra `"si"`; funcionaba por la normalización del JS, pero era frágil.
5. **No había validación**: `docentes = 0` producía una división por cero
   controlada de forma dudosa y los campos faltantes devolvían 500.
6. **La probabilidad se sobrescribía en el frontend**: para zona rural con
   matrícula ≥ 180 se forzaba un mínimo de 78 %.
7. **No había persistencia**: cada análisis se perdía al recargar.

## Escalado: corrección importante al Excel

El Excel indica “`StandardScaler` aplicado a las 41 columnas, luego
`predict`”. Los artefactos no respaldan esa afirmación:

- Los umbrales del árbol para `MATRICULA` están en unidades reales del censo
  (en el modelo original: 9.5 … 948; en el reentrenado hay **100 umbrales**
  entre 3.5 y 1958.5).
- Con el vector escalado las probabilidades dejan de tener sentido: en el
  modelo actual, una sede urbana de 950 estudiantes pasaría de **96.7 %** a
  **13.9 %**.
- Sin escalar, los tres casos de demo producen **0.0759, 0.6269 y 0.9672**,
  coherentes con la matrícula y la dotación.

Por eso el backend **no aplica el scaler**. `scaler_edutrack.pkl` se conserva
como referencia del entrenamiento. El notebook confirma que el árbol se entrenó
con `AD.fit(X_train, y_train)` y se predice con `AD.predict_proba(X_test)`, sin
escalar; los umbrales están en unidades reales.

## Clase 1

El Excel deja “significado de la clase 1: por confirmar”. El notebook lo define
como `REPORTA_FOCO = (N_FOCO > 0)`: **la sede reporta al menos un caso foco**
(TEA, trastorno del aprendizaje o condición psicosocial). La UI lo etiqueta como
“probabilidad de presencia de casos foco” y aclara que no es un diagnóstico.

## Reentrenamiento por resolución

El modelo original (`max_depth=6`, `min_samples_leaf=30`) tenía **51 hojas** y
solo **50 probabilidades distintas**, así que casi cualquier sede recibía el
mismo valor (por ejemplo, 48.77 % para cualquier matrícula entre 160 y 948).

Se reentrenó el mismo pipeline con `max_depth=24` y `min_samples_leaf=40`:

| Métrica | Original | Reentrenado |
|---|---|---|
| Hojas | 51 | **515** |
| Probabilidades distintas (prueba) | 50 | **339** |
| Umbrales de `MATRICULA` | 7 | **100** |
| Exactitud | 0.8674 | **0.8710** |
| F1 | 0.5160 | **0.5943** |
| AUC | 0.8736 | **0.8745** |

El script `scripts/reentrenar_modelo.py` reproduce las celdas 2–14 del notebook,
explora la rejilla de hiperparámetros y exporta los artefactos. Debe ejecutarse
con scikit-learn 1.6.1.

## Verificación

```bash
cd edutrack-backend
pytest -q
python scripts/verificar_modelo.py
```

Resultado actual: 52 pruebas aprobadas; los tres casos coinciden con el modelo
local con tolerancia `1e-9`.
