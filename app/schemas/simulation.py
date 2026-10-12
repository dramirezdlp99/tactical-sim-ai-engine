# Guardar en: tactical-sim-ai-engine/app/schemas/simulation.py (reemplaza el archivo actual)
#
# CAMBIO: se agregan 5 campos OPCIONALES con valor por defecto a la respuesta
# (predicted_defense, defense_scheme, clean_shot_probability, pass_effectiveness,
# model_info). Los campos anteriores conservan nombre y tipo, asi que el backend
# Java y el frontend actuales siguen funcionando sin cambios.

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

    # --- Campos nuevos del Modelo Predictivo + "Cuerpo Tecnico Rival" (todos opcionales) ---
    predicted_defense: List[PlayerPosition] = Field(
        default_factory=list,
        description="Posicion estimada de los defensores (tenis: del restador) DESPUES de reaccionar a la jugada.",
    )
    defense_scheme: str = Field(
        "", description="Esquema defensivo legible, ej. 'Hombre a hombre con ayuda en la pintura'."
    )
    clean_shot_probability: Optional[float] = Field(
        None, description="Baloncesto: probabilidad de que el portador anote si tira contra la defensa predicha."
    )
    pass_effectiveness: Optional[float] = Field(
        None, description="Baloncesto: probabilidad de completar el mejor pase disponible (o el pase recomendado)."
    )
    model_info: str = Field(
        "", description="Que modelo produjo la probabilidad y con que datos se entreno (o aviso de fallback)."
    )
