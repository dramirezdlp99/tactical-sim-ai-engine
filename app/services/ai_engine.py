"""
Guardar en: tactical-sim-ai-engine/app/services/ai_engine.py (reemplaza el archivo actual)

CAMBIO DE FONDO: la version anterior generaba una matriz aleatoria de
100x100 con numpy y devolvia el promedio. Por la Ley de los Grandes
Numeros, el promedio de 10,000 numeros uniformes(0,1) SIEMPRE converge
a ~0.5 sin importar los datos de entrada -- por eso el simulador daba
50% en absolutamente todos los casos, en ambos deportes. Ademas
"recommended_action" y "heat_map_coords" estaban fijos ("PASS_TO_WING"),
por eso aparecia identico en baloncesto y en tenis.

Esta version calcula todo con geometria real sobre las posiciones que
manda el frontend: distancia al aro, presion defensiva, apertura de
lineas de pase, distancia de alcance del restador, cercania a las
lineas del saque, etc. No es un modelo entrenado (eso requeriria un
dataset historico real que no existe en este proyecto) -- son formulas
deterministas y explicables, que es lo esperado para un motor "Modelo
Predictivo de Evaluacion Espacial" como lo describe tu propuesta de
casos de estudio.
"""

import asyncio
import math
import threading
from typing import List, Tuple

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


def _compute_open_space_grid(defenders: List[Tuple[str, Tuple[float, float]]], step: int = 20, top_n: int = 6) -> List[List[float]]:
    """Rejilla real de la cancha: devuelve los puntos con mayor distancia al defensor mas cercano
    (zonas de verdad abiertas), en vez de las dos coordenadas fijas que habia antes."""
    candidates = []
    for gx in range(0, 101, step):
        for gy in range(0, 101, step):
            nearest = min((_distance((gx, gy), d[1]) for d in defenders), default=999.0)
            candidates.append((nearest, [float(gx), float(gy)]))
    candidates.sort(key=lambda c: c[0], reverse=True)
    return [coords for _, coords in candidates[:top_n]]


def _simulate_basketball(data: dict) -> dict:
    attackers = [(p["player_id"], (p["x"], p["y"])) for p in data["team_home_positions"]]
    defenders = [(p["player_id"], (p["x"], p["y"])) for p in data["team_away_positions"]]
    ball = (data["ball_position"]["x"], data["ball_position"]["y"])

    if not attackers or not defenders:
        return {
            "success_probability": 0.0,
            "recommended_action": "DATOS_INSUFICIENTES",
            "heat_map_coords": [],
            "secondary_efficiency": 0.0,
            "risk_index": 1.0,
            "tactical_note": "No hay suficientes jugadores para calcular la jugada.",
            "ball_speed_kmh": None,
            "spin_rate_rpm": None,
        }

    # 1. Quien tiene el balon: el atacante mas cercano al balon
    holder_id, holder_pos = min(attackers, key=lambda a: _distance(a[1], ball))

    # 2. Presion defensiva real sobre quien tiene el balon
    nearest_def_to_holder = min(_distance(holder_pos, d[1]) for d in defenders)
    pressure = _clamp(1 - nearest_def_to_holder / DEFENSE_PRESSURE_RADIUS, 0.0, 1.0)

    # 3. Distancia real del portador al aro
    dist_to_hoop = _distance(holder_pos, BASKETBALL_HOOP)
    dist_factor = _clamp(1 - dist_to_hoop / COURT_DIAGONAL, 0.0, 1.0)

    # 4. Mejor companero con linea de pase realmente abierta (ningun defensor la corta) y buena posicion
    best_receiver = None
    best_receiver_score = -1.0
    for aid, apos in attackers:
        if aid == holder_id:
            continue
        nearest_def_to_receiver = min(_distance(apos, d[1]) for d in defenders)
        lane_clearance = min(_point_to_segment_distance(d[1], holder_pos, apos) for d in defenders)
        is_open = nearest_def_to_receiver > (DEFENSE_PRESSURE_RADIUS * 0.6) and lane_clearance > OPEN_LANE_THRESHOLD
        if not is_open:
            continue
        receiver_dist_to_hoop = _distance(apos, BASKETBALL_HOOP)
        score = _clamp(1 - receiver_dist_to_hoop / COURT_DIAGONAL, 0.0, 1.0)
        if score > best_receiver_score:
            best_receiver_score = score
            best_receiver = (aid, apos, receiver_dist_to_hoop)

    # 5. Decision tactica basada en datos reales, no en un valor fijo
    if dist_to_hoop <= SHOOT_RANGE and pressure <= 0.45:
        action = "TIRO_INMEDIATO"
        success = _clamp(0.35 + 0.45 * dist_factor - 0.30 * pressure, 0.05, 0.95)
        note = f"{holder_id} tiene tiro limpio a {dist_to_hoop:.1f} unidades del aro con presion baja ({pressure * 100:.0f}%)."
    elif best_receiver and (pressure > 0.55 or best_receiver[2] < dist_to_hoop - 10):
        action = f"PASE_A_{best_receiver[0]}"
        success = _clamp(0.30 + 0.50 * best_receiver_score - 0.15 * pressure, 0.05, 0.95)
        note = f"{holder_id} esta presionado ({pressure * 100:.0f}%); {best_receiver[0]} tiene mejor angulo y linea de pase abierta."
    elif dist_to_hoop > 45:
        action = "PENETRAR_AL_ARO"
        success = _clamp(0.25 + 0.35 * (1 - pressure) - 0.10 * (dist_to_hoop / COURT_DIAGONAL), 0.05, 0.90)
        note = f"{holder_id} esta lejos del aro ({dist_to_hoop:.1f} unidades) pero con espacio para penetrar."
    else:
        action = "REPOSICIONAR_Y_ESPERAR"
        success = 0.45
        note = f"{holder_id} no tiene una opcion clara: ni tiro limpio ni pase abierto disponible ahora mismo."

    # 6. Riesgo real de perdida de balon
    risk_index = _clamp(pressure * (0.7 if action == "REPOSICIONAR_Y_ESPERAR" else 0.4), 0.0, 1.0)

    # 7. Eficiencia de zona real (que tan buena es la posicion dentro del area de tiro efectivo)
    secondary_efficiency = _clamp(dist_factor * (1 - pressure * 0.5), 0.0, 1.0)

    return {
        "success_probability": round(success, 4),
        "recommended_action": action,
        "heat_map_coords": _compute_open_space_grid(defenders),
        "secondary_efficiency": round(secondary_efficiency, 4),
        "risk_index": round(risk_index, 4),
        "tactical_note": note,
        "ball_speed_kmh": None,
        "spin_rate_rpm": None,
    }


def _simulate_tennis(data: dict) -> dict:
    server = data["team_home_positions"][0]
    returner = data["team_away_positions"][0]
    target = (data["ball_position"]["x"], data["ball_position"]["y"])
    returner_pos = (returner["x"], returner["y"])
    server_pos = (server["x"], server["y"])

    # 1. Que tan lejos tiene que moverse el restador para llegar al punto de bote real
    reach_needed = _distance(returner_pos, target)
    reach_factor = _clamp(reach_needed / MAX_RETURNER_REACH, 0.0, 1.0)

    # 2. Que tan cerca de la banda cae el saque (mas cerca del borde = mas dificil de cubrir)
    edge_distance = min(target[0], 100 - target[0])
    edge_factor = _clamp(1 - edge_distance / 50.0, 0.0, 1.0)

    success = _clamp(0.30 + 0.50 * reach_factor + 0.20 * edge_factor, 0.05, 0.97)

    if reach_factor > 0.75:
        action = "ACE_PROBABLE"
        note = f"El restador necesita cubrir {reach_needed:.1f} unidades de cancha; muy dificil de alcanzar."
    elif reach_factor > 0.45:
        action = "SAQUE_GANADOR_PROBABLE"
        note = f"Saque efectivo: el restador llega con dificultad ({reach_needed:.1f} unidades de distancia)."
    else:
        action = "PUNTO_DISPUTADO"
        note = "El restador esta bien posicionado para este saque; se espera un intercambio largo."

    # 3. Riesgo real de doble falta / error (mas cerca de la linea = mas riesgo de salir)
    risk_index = round(_clamp(edge_factor * 0.5, 0.0, 1.0), 4)

    # 4. Eficiencia del angulo de saque: que tan cerca esta el recorrido de la distancia "ideal" de un buen saque
    travel_distance = _distance(server_pos, target)
    secondary_efficiency = round(_clamp(1 - abs(travel_distance - IDEAL_SERVE_TRAVEL) / IDEAL_SERVE_TRAVEL, 0.0, 1.0), 4)

    # 5. Velocidad/spin estimados con una aproximacion fisica simple a partir de la geometria
    #    (NO es una medicion Hawk-Eye real, es una estimacion proporcional a la distancia recorrida)
    ball_speed_kmh = round(140 + travel_distance * 0.9, 1)
    spin_rate_rpm = round(1800 + edge_factor * 1400, 0)

    return {
        "success_probability": round(success, 4),
        "recommended_action": action,
        "heat_map_coords": [[target[0], target[1]], [returner_pos[0], returner_pos[1]]],
        "secondary_efficiency": secondary_efficiency,
        "risk_index": risk_index,
        "tactical_note": note,
        "ball_speed_kmh": ball_speed_kmh,
        "spin_rate_rpm": spin_rate_rpm,
    }


def _run_simulation_sync(data: dict) -> dict:
    """
    Funcion sincrona (CPU-bound, aunque liviana) que corre en un HILO
    SECUNDARIO del thread pool, nunca en el event loop principal de
    FastAPI (ver AIEngineService.process_simulation_async). Se detecta
    el deporte por la cantidad de jugadores del equipo local: 1 = tenis
    (sacador vs restador), varios = baloncesto (5v5).
    """
    is_tennis = len(data.get("team_home_positions", [])) == 1

    result = _simulate_tennis(data) if is_tennis else _simulate_basketball(data)
    # Nombre real del hilo del pool -- prueba tangible de que esto NO
    # corre en el hilo principal (Main Thread) de FastAPI.
    result["execution_thread"] = threading.current_thread().name
    return result


class AIEngineService:
    async def process_simulation_async(self, simulation_data: dict) -> dict:
        """
        Delega el calculo geometrico a un hilo secundario via
        asyncio.to_thread, liberando el event loop principal para
        seguir aceptando otras peticiones concurrentes mientras se
        calcula esta jugada.
        """
        return await asyncio.to_thread(_run_simulation_sync, simulation_data)


ai_engine_service = AIEngineService()