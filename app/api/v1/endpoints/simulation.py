from fastapi import APIRouter, status
from app.schemas.simulation import TacticalSimulationRequest, TacticalSimulationResponse
from app.services.ai_engine import ai_engine_service

router = APIRouter()

@router.post("/predict", response_model=TacticalSimulationResponse, status_code=status.HTTP_200_OK)
async def predict_tactical_outcome(request: TacticalSimulationRequest):
    """
    Endpoint asíncrono que procesa la simulación táctica.
    La lógica pesada es delegada a un hilo secundario para evitar bloqueos del Event Loop.
    """
    result = await ai_engine_service.process_simulation_async(request.model_dump())
    return TacticalSimulationResponse(**result)