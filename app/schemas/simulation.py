from pydantic import BaseModel, Field
from typing import List, Optional

class PlayerPosition(BaseModel):
    player_id: str
    x: float = Field(..., description="Coordenada X en la cancha (0-100)")
    y: float = Field(..., description="Coordenada Y en la cancha (0-100)")

class TacticalSimulationRequest(BaseModel):
    match_id: str
    team_home_positions: List[PlayerPosition]
    team_away_positions: List[PlayerPosition]
    ball_position: PlayerPosition

class TacticalSimulationResponse(BaseModel):
    success_probability: float
    recommended_action: str
    heat_map_coords: List[List[float]]
    execution_thread: str = "WorkerThread-Async"