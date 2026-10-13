from fastapi import APIRouter
from app.api.v1.endpoints import simulation, model_info

api_router = APIRouter()
api_router.include_router(simulation.router, prefix="/simulation", tags=["Tactical Simulation"])
api_router.include_router(model_info.router, prefix="/model", tags=["Modelo predictivo"])
