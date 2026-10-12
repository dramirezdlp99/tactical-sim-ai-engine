"""
Guardar en: tactical-sim-ai-engine/app/services/ai_engine.py (reemplaza el archivo actual)

HISTORIA: la primera version devolvia el promedio de una matriz aleatoria
(siempre ~50%). La segunda calculaba todo con geometria y formulas fijas.

AHORA (esta version), la probabilidad de exito sale de un MODELO ENTRENADO
(regresion logistica, ver app/ml/ y docs/MODELO.md) y el rival REACCIONA:

    posiciones del entrenador
        -> respuesta defensiva ("Cuerpo Tecnico Rival", defense_response.py)
        -> caracteristicas sobre la defensa PREDICHA (app/ml/features.py)
        -> modelo entrenado (trained_model.py)
        -> success_probability
        -> promedio de miles de variantes Monte Carlo (monte_carlo.py)

Lo que NO cambia: la accion recomendada, la nota tactica, el mapa de calor, el
indice de eficiencia secundaria y el riesgo base siguen saliendo de la logica
geometrica de siempre, calculada sobre las posiciones dibujadas por el
entrenador. Asi la recomendacion es estable y explicable, y el modelo
cuantifica "que tan bien funciona esa jugada contra la defensa que reacciona".

FALLBACK SEGURO: si no hay pesos entrenados (archivo ausente/corrupto) o algo
falla al predecir, success_probability vuelve a las formulas geometricas
anteriores, EXACTAMENTE como antes. El motor nunca se cae.

IMPORTANTE (monte_carlo.py): `_simulate_basketball` y `_simulate_tennis`
siguen a nivel de modulo con la firma dict -> dict, porque monte_carlo las
busca por NOMBRE. Cada una tiene una version ligera `<nombre>_prob_only` que
solo calcula la probabilidad (sin mapa de calor, notas ni esquemas); las
iteraciones Monte Carlo usan esa para ir mas rapido.
"""

import asyncio
import math
import threading
from typing import Dict, List, Optional, Tuple

from app.ml import features as F
from app.services import trained_model
from app.services.defense_response import predict_basketball_defense, predict_tennis_returner

# Se vuelven a exponer aqui con los mismos nombres de siempre (viven en geometry.py
# para evitar imports circulares).
from app.services.geometry import (  # noqa: F401
    BASKETBALL_HOOP,
    COURT_DIAGONAL,
    DEFENSE_PRESSURE_RADIUS,
    IDEAL_SERVE_TRAVEL,
    MAX_RETURNER_REACH,
    OPEN_LANE_THRESHOLD,
    SHOOT_RANGE,
    _clamp,
    _distance,
    _point_to_segment_distance,
)
from app.services.monte_carlo import run_monte_carlo

# Cuando la mejor decision es "reposicionar y esperar" no hay una accion
# concreta que evaluar; se estima el valor de esperar como la mejor opcion
# disponible (tiro, pase o penetracion) con este descuento por perder tiempo
# de posesion. 0.85 es una eleccion de diseño, no un valor medido.
WAIT_DISCOUNT = 0.85


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


def _predicted_defense_payload(reaction) -> List[Dict]:
    return [{"player_id": pid, "x": round(pos[0], 2), "y": round(pos[1], 2)} for pid, pos in reaction.positions]


# ---------------------------------------------------------------------------
# BALONCESTO
# ---------------------------------------------------------------------------
def _basketball_geometric(attackers, defenders, ball, light: bool) -> dict:
    """Logica geometrica de siempre (accion, nota, riesgo, eficiencia) sobre las posiciones
    dibujadas. Tambien calcula `success_geo`, la probabilidad de las formulas antiguas, que
    sirve de respaldo si el modelo no esta disponible."""
    # 1. Quien tiene el balon: el atacante mas cercano al balon
    holder_idx = F.basketball_holder_index([a[1] for a in attackers], ball)
    holder_id, holder_pos = attackers[holder_idx]

    # 2. Presion defensiva real sobre quien tiene el balon
    nearest_def_to_holder = min(_distance(holder_pos, d[1]) for d in defenders)
    pressure = _clamp(1 - nearest_def_to_holder / DEFENSE_PRESSURE_RADIUS, 0.0, 1.0)

    # 3. Distancia real del portador al aro
    dist_to_hoop = _distance(holder_pos, BASKETBALL_HOOP)
    dist_factor = _clamp(1 - dist_to_hoop / COURT_DIAGONAL, 0.0, 1.0)

    # 4. Mejor companero con linea de pase realmente abierta (ningun defensor la corta) y buena posicion
    best_receiver = None
    best_receiver_score = -1.0
    for idx, (aid, apos) in enumerate(attackers):
        if idx == holder_idx:
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
            best_receiver = (aid, apos, receiver_dist_to_hoop, idx)

    # 5. Decision tactica basada en datos reales, no en un valor fijo
    note = ""
    receiver_idx = None
    if dist_to_hoop <= SHOOT_RANGE and pressure <= 0.45:
        action = "TIRO_INMEDIATO"
        success = _clamp(0.35 + 0.45 * dist_factor - 0.30 * pressure, 0.05, 0.95)
        if not light:
            note = f"{holder_id} tiene tiro limpio a {dist_to_hoop:.1f} unidades del aro con presion baja ({pressure * 100:.0f}%)."
    elif best_receiver and (pressure > 0.55 or best_receiver[2] < dist_to_hoop - 10):
        action = f"PASE_A_{best_receiver[0]}"
        receiver_idx = best_receiver[3]
        success = _clamp(0.30 + 0.50 * best_receiver_score - 0.15 * pressure, 0.05, 0.95)
        if not light:
            note = f"{holder_id} esta presionado ({pressure * 100:.0f}%); {best_receiver[0]} tiene mejor angulo y linea de pase abierta."
    elif dist_to_hoop > 45:
        action = "PENETRAR_AL_ARO"
        success = _clamp(0.25 + 0.35 * (1 - pressure) - 0.10 * (dist_to_hoop / COURT_DIAGONAL), 0.05, 0.90)
        if not light:
            note = f"{holder_id} esta lejos del aro ({dist_to_hoop:.1f} unidades) pero con espacio para penetrar."
    else:
        action = "REPOSICIONAR_Y_ESPERAR"
        success = 0.45
        if not light:
            note = f"{holder_id} no tiene una opcion clara: ni tiro limpio ni pase abierto disponible ahora mismo."

    # 6. Riesgo real de perdida de balon
    risk_index = _clamp(pressure * (0.7 if action == "REPOSICIONAR_Y_ESPERAR" else 0.4), 0.0, 1.0)

    # 7. Eficiencia de zona real (que tan buena es la posicion dentro del area de tiro efectivo)
    secondary_efficiency = _clamp(dist_factor * (1 - pressure * 0.5), 0.0, 1.0)

    return {
        "holder_idx": holder_idx, "action": action, "receiver_idx": receiver_idx,
        "success_geo": success, "note": note, "risk_index": risk_index,
        "secondary_efficiency": secondary_efficiency,
    }


def _basketball_model_eval(model, att_xy, holder_idx, def_xy, action, receiver_idx, full: bool):
    """Evalua la jugada elegida contra la defensa PREDICHA con el modelo entrenado.

    Devuelve (success, shot_p, pass_p). En modo ligero solo calcula lo que la
    accion elegida necesita; en modo completo calcula tambien tiro limpio y
    mejor pase para devolverlos en la respuesta.
    """
    holder = att_xy[holder_idx]
    shot_p = pass_p = drive_p = None

    if action == "TIRO_INMEDIATO":
        shot_p = model.predict("shot", F.basketball_features("shot", holder, def_xy))
        success = shot_p
    elif action.startswith("PASE_A_"):
        pass_p = model.predict("pass", F.basketball_features("pass", holder, def_xy, att_xy[receiver_idx]))
        success = pass_p
    elif action == "PENETRAR_AL_ARO":
        drive_p = model.predict("drive", F.basketball_features("drive", holder, def_xy))
        success = drive_p
    else:  # REPOSICIONAR_Y_ESPERAR: valor de la mejor opcion disponible, con descuento por esperar
        shot_p = model.predict("shot", F.basketball_features("shot", holder, def_xy))
        drive_p = model.predict("drive", F.basketball_features("drive", holder, def_xy))
        pass_p = _best_pass(model, att_xy, holder_idx, def_xy)
        best = max(v for v in (shot_p, drive_p, pass_p) if v is not None)
        success = WAIT_DISCOUNT * best

    if full:
        if shot_p is None:
            shot_p = model.predict("shot", F.basketball_features("shot", holder, def_xy))
        if pass_p is None:
            pass_p = _best_pass(model, att_xy, holder_idx, def_xy)
    return success, shot_p, pass_p


def _best_pass(model, att_xy, holder_idx, def_xy) -> Optional[float]:
    """Mejor probabilidad de completar un pase a cualquier companero (None si no hay companeros)."""
    holder = att_xy[holder_idx]
    best = None
    for i, a in enumerate(att_xy):
        if i == holder_idx:
            continue
        p = model.predict("pass", F.basketball_features("pass", holder, def_xy, a))
        if best is None or p > best:
            best = p
    return best


def _basketball_core(data: dict, light: bool) -> dict:
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
            "predicted_defense": [],
            "defense_scheme": "",
            "clean_shot_probability": None,
            "pass_effectiveness": None,
            "model_info": trained_model.model_info("basketball"),
        }

    geo = _basketball_geometric(attackers, defenders, ball, light)
    success = geo["success_geo"]
    risk_index = geo["risk_index"]
    note = geo["note"]
    action = geo["action"]
    scheme = ""
    predicted_defense: List[Dict] = []
    shot_p = pass_p = None

    model = trained_model.load_model("basketball")
    if model is not None:
        try:
            reaction = predict_basketball_defense(attackers, defenders, ball, geo["holder_idx"])
            att_xy = [a[1] for a in attackers]
            def_xy = [p[1] for p in reaction.positions]
            m_success, shot_p, pass_p = _basketball_model_eval(
                model, att_xy, geo["holder_idx"], def_xy, action, geo["receiver_idx"], full=not light)
            success = m_success
            scheme = reaction.scheme
            if not light:
                predicted_defense = _predicted_defense_payload(reaction)
                if action.startswith("PASE_A_") and pass_p is not None:
                    risk_index = _clamp(1.0 - pass_p, 0.0, 1.0)   # riesgo de perdida = pase no completado
                note = (f"{note} Tras la reaccion del rival ({scheme}), con las posiciones exactas que dibujaste "
                        f"el modelo estima {success * 100:.0f}% de exito.")
        except Exception:
            # Cualquier fallo del modelo o de la respuesta defensiva: volver a las formulas geometricas.
            success, shot_p, pass_p, scheme, predicted_defense = geo["success_geo"], None, None, "", []
            risk_index, note = geo["risk_index"], geo["note"]
            model = None

    if light:
        return {"success_probability": success}

    return {
        "success_probability": round(success, 4),
        "recommended_action": action,
        "heat_map_coords": _compute_open_space_grid(defenders),
        "secondary_efficiency": round(geo["secondary_efficiency"], 4),
        "risk_index": round(risk_index, 4),
        "tactical_note": note,
        "ball_speed_kmh": None,
        "spin_rate_rpm": None,
        "predicted_defense": predicted_defense,
        "defense_scheme": scheme,
        "clean_shot_probability": None if shot_p is None else round(shot_p, 4),
        "pass_effectiveness": None if pass_p is None else round(pass_p, 4),
        "model_info": trained_model.model_info("basketball") if model is not None
        else "Formulas geometricas deterministas (modelo entrenado no disponible)",
    }


def _simulate_basketball(data: dict) -> dict:
    """Evaluacion completa de baloncesto (dict -> dict). monte_carlo.py la resuelve por nombre."""
    return _basketball_core(data, light=False)


def _simulate_basketball_prob_only(data: dict) -> dict:
    """Version ligera para Monte Carlo: solo {"success_probability": ...}, sin mapa de calor ni notas."""
    return _basketball_core(data, light=True)


# ---------------------------------------------------------------------------
# TENIS
# ---------------------------------------------------------------------------
def _tennis_core(data: dict, light: bool) -> dict:
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

    # 3. Modelo entrenado sobre la posicion ANTICIPADA del restador (Cuerpo Tecnico Rival)
    scheme = ""
    predicted_defense: List[Dict] = []
    model = trained_model.load_model("tennis")
    if model is not None:
        try:
            reaction = predict_tennis_returner(server_pos, (returner["player_id"], returner_pos), target)
            new_pos = reaction.positions[0][1]
            success = model.predict("serve", F.tennis_serve_features(server_pos, new_pos, target))
            scheme = reaction.scheme
            predicted_defense = _predicted_defense_payload(reaction)
        except Exception:
            success = _clamp(0.30 + 0.50 * reach_factor + 0.20 * edge_factor, 0.05, 0.97)
            scheme, predicted_defense, model = "", [], None

    if light:
        return {"success_probability": success}

    if reach_factor > 0.75:
        action = "ACE_PROBABLE"
        note = f"El restador necesita cubrir {reach_needed:.1f} unidades de cancha; muy dificil de alcanzar."
    elif reach_factor > 0.45:
        action = "SAQUE_GANADOR_PROBABLE"
        note = f"Saque efectivo: el restador llega con dificultad ({reach_needed:.1f} unidades de distancia)."
    else:
        action = "PUNTO_DISPUTADO"
        note = "El restador esta bien posicionado para este saque; se espera un intercambio largo."
    if model is not None:
        note = f"{note} Con la anticipacion del restador y las posiciones exactas dibujadas, el modelo estima {success * 100:.0f}% de ganar el punto al saque."

    # 4. Riesgo real de doble falta / error (mas cerca de la linea = mas riesgo de salir)
    risk_index = round(_clamp(edge_factor * 0.5, 0.0, 1.0), 4)

    # 5. Eficiencia del angulo de saque: que tan cerca esta el recorrido de la distancia "ideal" de un buen saque
    travel_distance = _distance(server_pos, target)
    secondary_efficiency = round(_clamp(1 - abs(travel_distance - IDEAL_SERVE_TRAVEL) / IDEAL_SERVE_TRAVEL, 0.0, 1.0), 4)

    # 6. Velocidad/spin estimados con una aproximacion fisica simple a partir de la geometria
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
        "predicted_defense": predicted_defense,
        "defense_scheme": scheme,
        "clean_shot_probability": None,
        "pass_effectiveness": None,
        "model_info": trained_model.model_info("tennis") if model is not None
        else "Formulas geometricas deterministas (modelo entrenado no disponible)",
    }


def _simulate_tennis(data: dict) -> dict:
    """Evaluacion completa de tenis (dict -> dict). monte_carlo.py la resuelve por nombre."""
    return _tennis_core(data, light=False)


def _simulate_tennis_prob_only(data: dict) -> dict:
    """Version ligera para Monte Carlo: solo {"success_probability": ...}."""
    return _tennis_core(data, light=True)


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
        1) Motor de evaluacion espacial determinista (accion recomendada,
           nota tactica, mapa de calor): corre en un hilo secundario via
           asyncio.to_thread, liberando el event loop principal para
           seguir aceptando otras peticiones concurrentes mientras se
           calcula esta jugada.

        2) CAMBIO (brecha #2 del Caso de Estudio 2, seccion 2.2): la
           probabilidad de exito final ya NO es ese unico calculo
           determinista. Se ejecuta ademas una simulacion Monte Carlo de
           miles de variantes aleatorias de la misma jugada, repartida en
           PROCESOS reales del sistema operativo que comparten un bloque
           de memoria (ver monte_carlo.py). success_probability pasa a
           ser el promedio real de esas miles de iteraciones, y se deja
           constancia de ello en tactical_note y en execution_thread
           (antes siempre decia "WorkerThread-Async" o el nombre de un
           solo hilo; ahora muestra cuantos procesos y cuantas
           iteraciones realmente se ejecutaron, para poder verificarlo).
        """
        base_result = await asyncio.to_thread(_run_simulation_sync, simulation_data)

        is_tennis = len(simulation_data.get("team_home_positions", [])) == 1
        deterministic_fn_name = "_simulate_tennis" if is_tennis else "_simulate_basketball"

        mc_result = await asyncio.to_thread(run_monte_carlo, simulation_data, deterministic_fn_name)

        base_result["success_probability"] = mc_result["success_probability_mc"]
        base_result["tactical_note"] = (
            f"{base_result['tactical_note']} "
            f"Al promediar {mc_result['iterations']} variaciones realistas de las posiciones "
            f"(Monte Carlo en {mc_result['workers']} procesos) la probabilidad final es "
            f"{mc_result['success_probability_mc'] * 100:.0f}% (variacion de ±{mc_result['success_std_dev'] * 100:.1f} puntos)."
        ).strip()
        base_result["execution_thread"] = (
            f"MultiProcess-SharedMemory(workers={mc_result['workers']}, "
            f"iterations={mc_result['iterations']})"
        )
        return base_result


ai_engine_service = AIEngineService()
