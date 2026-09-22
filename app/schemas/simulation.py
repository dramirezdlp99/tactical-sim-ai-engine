# Guardar en: tactical-sim-ai-engine/app/schemas/simulation.py (reemplaza el archivo actual)

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

    # CAMBIO: campos nuevos para que la respuesta sea mas completa y no
    # se sienta repetitiva (antes success_probability era el unico dato
    # real, y el frontend inventaba el resto multiplicando por ratios fijos).
    secondary_efficiency: float = Field(
        0.0, description="Baloncesto: eficiencia de zona. Tenis: limpieza del angulo de saque."
    )
    risk_index: float = Field(
        0.0, description="Baloncesto: riesgo de perdida de balon. Tenis: riesgo de doble falta."
    )
    tactical_note: str = Field(
        "", description="Explicacion en lenguaje natural de por que se calculo este resultado."
    )
    ball_speed_kmh: Optional[float] = Field(
        None, description="Solo tenis: velocidad estimada del saque."
    )
    spin_rate_rpm: Optional[float] = Field(
        None, description="Solo tenis: spin estimado del saque."
    )