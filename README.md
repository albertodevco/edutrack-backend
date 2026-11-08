# EduTrack Backend

API REST desarrollada en Flask para la inferencia del modelo predictivo del proyecto EduTrack.

## 📋 Requisitos
- Python 3.9+
- Dependencias listadas en `requirements.txt`

## 🚀 Instalación y Ejecución

1. Crear un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Linux/macOS:
   source venv/bin/activate
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Iniciar el servidor:
   ```bash
   python app.py
   ```
   La aplicación se ejecutará por defecto en `http://localhost:5000`.

## 📡 Endpoints

### `POST /predecir`
Recibe las variables de la institución y retorna la probabilidad calculada por el modelo.

**Body (JSON):**
```json
{
  "matricula": 500,
  "docentes": 25,
  "orientadores": 2,
  "niveles": 3,
  "jornadas": 1,
  "zona": "rural",
  "bilingue": "no",
  "etno": "no"
}
```

**Respuesta:**
```json
{
  "probabilidad": 0.42
}
```
