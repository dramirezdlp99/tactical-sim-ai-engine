from fastapi import APIRouter
from app.api.v1.endpoints import simulation

api_router = APIRouter()
api_router.include_router(simulation.router, prefix="/simulation", tags=["Tactical Simulation"])