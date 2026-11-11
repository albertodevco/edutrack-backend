# Publicar EduTrack en GitHub

EduTrack se publica en **dos repositorios independientes**, que ya existen y ya
tienen su remoto configurado:

| Repositorio local | Remoto |
|---|---|
| `edutrack-backend` | <https://github.com/albertodevco/edutrack-backend> |
| `edutrack-frontend` | <https://github.com/albertodevco/edutrack-frontend> |

Los cambios de esta versión ya están en el árbol de trabajo de cada repositorio,
listos para `git add` + `git commit` + `git push`. No hay que crear repositorios
nuevos.

---

## 0. Qué cambia en cada repositorio

### `edutrack-backend`

```text
app.py                     endpoints /, /salud, /predecir
schemas.py                 validación Pydantic (422, extra="forbid")
model_service.py           vector de 41 columnas, sin scaler
requirements.txt           scikit-learn==1.6.1, numpy==2.1.3
requirements-dev.txt       pytest
runtime.txt                python-3.12.7
.python-version            3.12.7
.env.example               variables de entorno documentadas
.gitattributes             normalización de saltos de línea
.github/workflows/tests.yml  CI: 50 pruebas en cada push
tests/test_api.py          50 pruebas de contrato y regresión
scripts/verificar_modelo.py  verificación reproducible
docs/DIAGNOSTICO.md        código anterior vs. modelo real
docs/PUBLICAR_EN_GITHUB.md esta guía
```

### `edutrack-frontend`

```text
index.html                 formulario de 9 campos, resultados e historial
login.html                 Supabase Auth con config.js
script.js                  sin probabilidad forzada, guardado e historial
style.css                  switches, parámetros, historial, responsive
config.js                  configuración real (clave publicable)
config.example.js          plantilla
.gitattributes
supabase/migrations/001_analisis_sede.sql   tabla + RLS
```

> Los `.pkl` del backend son pequeños (13 KB en total) y **deben** estar en el
> repositorio: Render los necesita al arrancar. No los ignores.

---

## 1. Revisar antes de subir

```powershell
cd "C:\Users\Admin\Documents\Inteligencia Artificial\edutrack-backend"
git status
git diff --stat

cd "..\edutrack-frontend"
git status
git diff --stat
```

Comprueba que **no** aparezcan `.env`, `venv/`, `__pycache__/` ni `.pytest_cache/`.
Si aparece un `.env` real, detente y bórralo del índice antes de continuar.

---

## 2. Subir el backend

```powershell
cd "C:\Users\Admin\Documents\Inteligencia Artificial\edutrack-backend"

git add -A
git commit -m "feat: validación Pydantic, vector de 41 columnas y modelo sin escalado"

# Si el remoto tiene commits nuevos (por ejemplo, ediciones desde GitHub):
git pull --rebase origin main

git push origin main
```

## 3. Subir el frontend

```powershell
cd "C:\Users\Admin\Documents\Inteligencia Artificial\edutrack-frontend"

git add -A
git commit -m "feat: formulario de 9 campos, resultados reales del modelo e historial Supabase"

git pull --rebase origin main
git push origin main
```

### Autenticación

- **Git Credential Manager (recomendado):** la primera vez se abre el navegador
  y autorizas con tu cuenta de GitHub.
- **Token personal:** GitHub → *Settings* → *Developer settings* → *Personal
  access tokens* → *Tokens (classic)* → **Generate new token** con permiso
  `repo` (y `workflow` para que se ejecute el CI). Úsalo como contraseña.

### Verificar

```powershell
git log --oneline -3
git remote -v
git status          # debe decir "nothing to commit, working tree clean"
```

En GitHub, pestaña **Actions** del backend: el workflow *Pruebas del backend*
debe pasar en verde (50 pruebas + verificación del modelo).

---

## 4. Desplegar el backend en Render

1. <https://dashboard.render.com> → **New +** → **Web Service**.
2. Conecta el repositorio `edutrack-backend`.
3. Configuración:

   | Campo | Valor |
   |---|---|
   | Branch | `main` |
   | Root Directory | *(vacío)* |
   | Runtime | `Python 3` |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `gunicorn app:app` |
   | Instance Type | Free |

4. **Environment Variables** (opcionales):

   | Variable | Valor |
   |---|---|
   | `ALLOWED_ORIGINS` | `https://TU-FRONTEND.vercel.app` o `*` para demo |
   | `LOG_LEVEL` | `INFO` |

5. **Create Web Service**. El primer deploy tarda unos minutos.
6. Verifica:

   ```bash
   curl -s https://TU-SERVICIO.onrender.com/salud
   ```

   Debe responder `"estado":"ok"`, `"n_features":41`, `"clases":[0,1]` y
   `"escalado_en_inferencia":false`.

> `runtime.txt` / `.python-version` fijan Python 3.12 y `requirements.txt` fija
> `scikit-learn==1.6.1`, la versión con la que se serializó el modelo.

---

## 5. Crear la tabla en Supabase

1. Supabase → proyecto → **SQL Editor** → **New query**.
2. Pega el contenido de
   `edutrack-frontend/supabase/migrations/001_analisis_sede.sql`.
3. **Run**. Debe terminar sin errores.
4. Verifica en **Table Editor** que existe `analisis_sede` y en
   **Authentication → Policies** que están las dos políticas por usuario.

---

## 6. Publicar el frontend (Vercel)

1. <https://vercel.com/new> → importa el repositorio `edutrack-frontend`.
2. **Root Directory:** vacío (el repo ya es la carpeta del sitio).
3. **Framework Preset:** `Other`. Sin build.
4. **Deploy**.
5. Edita `config.js` con la URL real del backend y haz commit (Vercel
   redepliega solo):

   ```js
   window.EDUTRACK_CONFIG = {
     SUPABASE_URL: 'https://TU-PROYECTO.supabase.co',
     SUPABASE_ANON_KEY: 'TU_CLAVE_PUBLICABLE',
     API_URL: 'https://TU-SERVICIO.onrender.com',
     TABLA_ANALISIS: 'analisis_sede'
   };
   ```

6. Si restringiste `ALLOWED_ORIGINS` en Render, añade el dominio de Vercel y
   guarda (Render reinicia).

### GitHub Pages (alternativa)

*Settings* → *Pages* → *Source*: `Deploy from a branch` → rama `main` →
carpeta `/ (root)`. URL: `https://albertodevco.github.io/edutrack-frontend/`.

---

## 7. Flujo de trabajo diario

```powershell
# backend
cd "C:\Users\Admin\Documents\Inteligencia Artificial\edutrack-backend"
git add -A
git commit -m "fix: descripción breve"
git push

# frontend
cd "..\edutrack-frontend"
git add -A
git commit -m "feat: descripción breve"
git push
```

Cada push a `main` ejecuta el CI del backend y redespliega Render / Vercel.

---

## 8. Errores frecuentes

| Error | Causa | Solución |
|---|---|---|
| `failed to push some refs` / `rejected` | el remoto tiene commits nuevos | `git pull --rebase origin main` y luego `git push` |
| `Authentication failed` | token o credenciales inválidas | reautentica con Git Credential Manager o genera un token con permiso `repo` |
| `remote origin already exists` | el remoto ya está configurado | `git remote set-url origin <URL>` |
| Render: `ModuleNotFoundError` | Root Directory incorrecto | en este esquema debe estar vacío |
| Render falla al cargar el `.pkl` | versión de sklearn distinta | confirma `scikit-learn==1.6.1` |
| Frontend: error de CORS | origen no permitido | ajusta `ALLOWED_ORIGINS` o usa `*` |
| Historial vacío / 401 | falta la tabla o las políticas RLS | ejecuta el SQL del paso 5 |
| `InconsistentVersionWarning` en local | tu sklearn no es 1.6.1 | usa el venv con `requirements.txt`; en Render no aparece |

---

## 9. Seguridad

- `config.js` contiene la **clave publicable** de Supabase: es segura en el
  navegador **solo si RLS está activo**, y lo está en la migración incluida.
- **Nunca** subas la clave `service_role`, contraseñas de base de datos ni un
  `.env` real. Ya están en `.gitignore`.
- Si pegas una clave privada por error: revócala en Supabase y limpia el
  historial (`git filter-repo`) antes de hacer público el repositorio.

---

## 10. Opcional

- Añade `LICENSE` desde *Add file* → *Create new file* → `LICENSE`.
- En **About** (⚙️) del repositorio: descripción y temas
  (`machine-learning`, `flask`, `scikit-learn`, `supabase`, `education`).
