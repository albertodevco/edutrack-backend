# Diagnóstico: código anterior vs. parámetros reales del modelo

Comparación solicitada en la hoja **Diagnóstico** de
`edutrack_parametros_modelo.xlsx`, ya corregida en el código actual.

## Estado final

| Parámetro | ¿Está en el formulario? | ¿Llega al payload? | ¿Llega al modelo? | Observación |
|---|---|---|---|---|
| `MATRICULA` | Sí | Sí | Sí | Entero 0 – 100 000 |
| `ZONA_RURAL` | Sí (select Rural/Urbana) | Sí (`zona`) | Sí (1/0) | Segunda variable en importancia (10,8 %) |
| `N_DOCENTES` | Sí | Sí | Sí | Entero ≥ 1; divisor de `EST_POR_DOCENTE` |
| `N_ORIENTADORES` | Sí | Sí | Sí | Entero ≥ 0 |
| `ETNOEDUC_SI` | Sí (switch) | Sí (booleano) | Sí (1/0) | Peso 1,07 % |
| `N_NIVELES` | Sí | Sí | Sí | Entero 1 – 6 |
| `N_JORNADAS` | Sí | Sí | Sí | Entero 1 – 5; importancia 0 pero columna obligatoria |
| `BILINGUE_SI` | Sí (switch) | Sí (booleano) | Sí (1/0) | Importancia 0 pero columna obligatoria |
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

- Los umbrales del árbol para `MATRICULA` son **9.5, 21.5, 55.5, 159.5, 249.5 y
  948**, es decir, unidades reales del censo.
- Con el vector escalado, la probabilidad de los casos de demo queda en
  **0.3125** para casi cualquier sede (la matrícula escalada nunca supera 159.5).
- Sin escalar, los tres casos producen 0.1171, 0.4335 y 0.7785, coherentes con
  la matrícula y la dotación.

Por eso el backend **no aplica el scaler**. `scaler_edutrack.pkl` se conserva
como referencia del entrenamiento. Si el notebook confirma que el escalado sí
formaba parte del pipeline, habría que reentrenar o re-serializar el modelo; no
basta con aplicar `scaler.transform` al árbol actual.

## Clase 1

El Excel deja “significado de la clase 1: por confirmar”. La evidencia
disponible:

- La probabilidad crece con la matrícula (0.13 → 0.49 → 0.78).
- Aumenta ligeramente cuando hay orientadores.
- El árbol usa matrícula, zona, docentes, orientadores y departamento.

Eso es consistente con **“la sede registra al menos un caso foco”**, no con
“riesgo de subregistro”. La UI etiqueta la salida como “probabilidad de
presencia de casos foco” y aclara que no es un diagnóstico. **Pendiente:**
confirmarlo contra la definición del target en el notebook de entrenamiento.

## Verificación

```bash
cd edutrack-backend
pytest -q
python scripts/verificar_modelo.py
```

Resultado actual: 50 pruebas aprobadas; los tres casos coinciden con el modelo
local con tolerancia `1e-9`.
