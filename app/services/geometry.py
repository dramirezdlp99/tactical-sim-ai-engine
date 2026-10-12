"""
Guardar en: tactical-sim-ai-engine/app/services/geometry.py

Geometria basica compartida del AI Engine (Python puro, sin numpy).

Estas funciones y constantes vivian dentro de ai_engine.py. Se movieron a
un modulo propio para que ai_engine.py, defense_response.py y el paquete
app/ml/ las usen SIN importarse entre si en circulo (un import circular
haria que el servidor no arrancara). ai_engine.py las vuelve a exponer con
los mismos nombres de siempre, asi que nada de lo que ya las usaba cambia.

Sistema de coordenadas: todo va de 0 a 100 (porcentaje de la cancha), igual
que el frontend.
"""

import math
from typing import Tuple

# ---- Constantes de cancha de baloncesto (mismo sistema 0-100 que usa el frontend) ----
BASKETBALL_HOOP = (50.0, 10.0)          # posicion aproximada del aro en el medio campo
DEFENSE_PRESSURE_RADIUS = 15.0          # dentro de este radio, un defensor presiona al portador
OPEN_LANE_THRESHOLD = 10.0              # distancia minima de un defensor a la linea de pase para considerarla "abierta"
SHOOT_RANGE = 24.0                      # distancia al aro para considerar tiro inmediato razonable
COURT_DIAGONAL = math.hypot(100, 100)   # ~141.42, para normalizar distancias a 0-1

# ---- Constantes de cancha de tenis ----
MAX_RETURNER_REACH = 28.0               # distancia (en % de cancha) que un restador de elite cubre comodamente
IDEAL_SERVE_TRAVEL = 55.0               # distancia aproximada de un buen saque profundo


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_to_segment_distance(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Distancia perpendicular de un punto p al segmento a-b (para saber si un defensor corta una linea de pase)."""
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return _distance(p, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj = (ax + t * dx, ay + t * dy)
    return _distance(p, proj)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
