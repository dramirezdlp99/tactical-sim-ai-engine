"""
Guardar en: tactical-sim-ai-engine/app/api/v1/endpoints/model_info.py

GET /api/v1/model/info -> metadatos y metricas de validacion de los modelos
entrenados. Es publico (no expone pesos) y lo consume la pagina /modelo del
frontend, que se regenera sola cada cierto tiempo (ISR).
"""

from fastapi import APIRouter

from app.services import model_summary

router = APIRouter()


@router.get("/info", summary="Metadatos y metricas del modelo predictivo")
def model_info():
    return model_summary.summary()
