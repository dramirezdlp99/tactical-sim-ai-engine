# 🧠 TacticAI — Motor de IA (FastAPI)

> **Predice la probabilidad de éxito de acciones tácticas con modelos entrenados y simulación Monte Carlo.**
> Caso de Estudio 2 · Programación Orientada a la Web

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy&logoColor=white)
![Tests](https://img.shields.io/badge/tests-43%20passing-brightgreen)

---

## 👨‍🎓 Autoría

| | |
|---|---|
| **Desarrollado por** | David Fernando Ramírez de la Parra |
| **Profesor** | Jhonatan Andrés Mideros Narváez |
| **Asignatura** | Programación Orientada a la Web |
| **Institución** | Universidad Cooperativa de Colombia |

---

## 🌐 Servicios

| Servicio | URL |
|---|---|
| 🧠 Motor de IA | https://tactical-sim-ai-engine.onrender.com |
| ❤️ Salud | https://tactical-sim-ai-engine.onrender.com/health |
| 📊 Info del modelo | https://tactical-sim-ai-engine.onrender.com/api/v1/model/info |
| ⚙️ Backend | https://tactical-sim-backend.onrender.com |
| 🖥️ Frontend | https://tactical-sim-frontend.vercel.app |

> ⏳ Render gratis: arranque en frío de ~50 s tras inactividad.

---

## 📚 Rol en el caso de estudio

Es la **capa de IA** de TacticAI, un servicio **independiente** detrás del backend. El navegador nunca lo llama directamente; el backend lo consulta con una **clave interna**. El cálculo se procesa en un **hilo secundario** para no bloquear el servidor (principio de clase: *la IA debe trabajar fuera del hilo principal*).

## 🧪 Cómo funciona el modelo

| Elemento | Detalle |
|---|---|
| Algoritmo | **Regresión logística** entrenada |
| Baloncesto | 3 cabezas: **tiro**, **pase**, **penetración** (drive) |
| Tenis | 1 cabeza: **saque** (serve) |
| Datos | **80 000 muestras sintéticas por deporte**, semilla 42 |
| Incertidumbre | **Monte Carlo** con σ = 2.5 |
| Defensa rival | **Basada en reglas** ("Cuerpo Técnico Rival") |

### 📈 Métricas

| Cabeza | AUC |
|---|---|
| Tiro | 0.61 |
| Pase | 0.81 |
| Penetración | 0.63 |
| Saque (tenis) | 0.64 |

Brier ≈ 0.17–0.21 · Error de calibración (ECE) ≈ 0.008.
Detalle completo en [`docs/MODELO.md`](docs/MODELO.md).

### ⚠️ Honestidad científica

- Los datos son **sintéticos** (generados por reglas), no partidos reales: el modelo aprende esas reglas, no la realidad.
- La respuesta defensiva está basada en reglas, no aprendida.
- Los AUC moderados reflejan el ruido deliberado del generador.

---

## 🔌 Endpoints

| Método | Ruta | Acceso | Descripción |
|---|---|---|---|
| GET | `/health` | 🌍 público | Estado del servicio |
| POST | `/api/v1/simulation/predict` | 🔑 clave interna | Predicción + Monte Carlo + defensa |
| GET | `/api/v1/model/info` | 🌍 público | Metadatos y métricas (**sin pesos**) |

Documentación interactiva: `/docs` (Swagger).

## 🔐 Seguridad entre servicios

Si la variable `AI_INTERNAL_KEY` está definida, `/api/v1/simulation/predict` exige la cabecera:

```
X-Internal-Key: <clave>
```

Sin ella, o con una incorrecta, responde **401**. Se compara con `secrets.compare_digest` (tiempo constante). Si la variable **no** está definida, no se exige clave (útil en local).
La misma clave debe configurarse en el **backend**. 🚫 Nunca se sube al repositorio.

---

## 🔧 Variables de entorno

| Variable | Obligatoria | Descripción |
|---|---|---|
| `AI_INTERNAL_KEY` | ➖ (recomendada en producción) | Clave compartida con el backend |
| `PORT` | ➖ | Puerto (Render lo inyecta) |

## 🚀 Uso local

Requisito: **Python 3.11**.

```powershell
git clone https://github.com/TU-USUARIO/tactical-sim-ai-engine.git
cd tactical-sim-ai-engine
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Abre http://localhost:8000/docs

Prueba rápida:

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/model/info
```

## 🧪 Pruebas

```powershell
pip install -r requirements-dev.txt
pytest
```

**43 pruebas**: `test_modelo`, `test_model_info` (5), `test_internal_auth` (5), entre otras. Se ejecutan en GitHub Actions con Python 3.11.

> 💡 `requirements-dev.txt` fija `httpx==0.27.2`, necesario para el `TestClient` de la versión de Starlette usada.

## 📂 Estructura

```
app/
├── api/v1/            # api.py y endpoints (simulation, model_info)
├── core/              # config, internal_auth (clave interna)
├── ml/                # dataset, features, models/*.json
└── services/          # ai_engine, trained_model, monte_carlo,
                       # defense_response, geometry, model_summary
docs/MODELO.md         # documentación del modelo
tests/                 # 43 pruebas
runtime.txt            # Python 3.11.9
```

Dependencias fijadas: `fastapi 0.110.0`, `uvicorn 0.28.0`, `pydantic 2.6.4`, `numpy 1.26.4`, `scipy 1.12.0`.

## ☁️ Despliegue (Render)

1. Crea un **Web Service** (Python) en Render apuntando a este repo.
2. Build: `pip install -r requirements.txt`
3. Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Define `AI_INTERNAL_KEY` (la misma que en el backend).
5. Healthcheck: `/health`.

## 🗺️ Próximos pasos

- Entrenar con datos reales de partidos.
- Defensa aprendida (en lugar de reglas).
- Más cabezas (tenis: devolución, golpes de fondo).

---

<p align="center">
Hecho por <b>David Fernando Ramírez de la Parra</b><br>
Profesor: Jhonatan Andrés Mideros Narváez · Programación Orientada a la Web · Universidad Cooperativa de Colombia
</p>
