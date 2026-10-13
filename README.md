# TacticAI — Motor de IA

Servicio FastAPI que simula jugadas de baloncesto y tenis: modelo logístico entrenado, Monte Carlo y respuesta defensiva. Más detalle del modelo en [`docs/MODELO.md`](docs/MODELO.md).

> Los datos de entrenamiento son **sintéticos** (generados por reglas con semilla fija). El proyecto demuestra la arquitectura; no está validado con partidos reales.

## Uso local

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (PowerShell)
pip install -r requirements.txt -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
python -m pytest -q                 # 38 pruebas
```

## Endpoints principales

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio |
| POST | `/api/v1/simulation/...` | Simulación táctica (la consume el backend) |
| GET | `/api/v1/model/info` | Resumen público del modelo (sin pesos), usado por la página `/modelo` del frontend |

## Despliegue

Render (Python 3.11.9, ver `runtime.txt`). `requirements-dev.txt` no se instala en producción.

## Nota sobre httpx

Las pruebas usan el `TestClient` de FastAPI, que con Starlette 0.36 necesita `httpx==0.27.2`. Está fijado en `requirements-dev.txt`.
