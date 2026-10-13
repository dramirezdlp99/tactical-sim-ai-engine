from fastapi import APIRouter, Depends
from app.api.v1.endpoints import simulation, model_info
from app.core.internal_auth import require_internal_key

api_router = APIRouter()
# La simulacion solo la puede pedir el backend (clave interna); el resumen del modelo es publico.
api_router.include_router(
    simulation.router, prefix="/simulation", tags=["Tactical Simulation"], dependencies=[Depends(require_internal_key)]
)
api_router.include_router(model_info.router, prefix="/model", tags=["Modelo predictivo"])
