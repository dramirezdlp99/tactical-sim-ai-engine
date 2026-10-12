"""
Guardar en: tactical-sim-ai-engine/app/ml/dataset.py

GENERADOR DE DATOS SINTETICOS (con semilla fija) PARA ENTRENAR EL MODELO.

HONESTIDAD ANTE TODO: este proyecto NO tiene datos historicos reales de
partidos (ni tracking de jugadores, ni resultados de tiros/pases/saques).
Por eso aqui se fabrican "partidos de mentira" de la siguiente forma:

  1. Se inventan situaciones (posiciones de atacantes, defensores, balon)
     con una variedad amplia: defensas pegadas, sueltas, ayudas en la
     pintura, jugadores al azar, etc.
  2. Cada situacion tiene una PROBABILIDAD VERDADERA de exito calculada con
     la "verdad de terreno" de este archivo: reglas basadas en conocimiento
     publico del deporte (porcentajes tipicos de tiro por distancia, efecto
     de tener un defensor encima, intercepcion de pases, tiempos de reaccion
     en el resto de un saque...). Cada constante lleva comentada su fuente
     aproximada. Son ORDENES DE MAGNITUD razonables, no estadisticas de una
     liga concreta.
  3. El resultado de cada intento se SORTEA (Bernoulli): sale 1 (exito) o 0
     (fallo) con esa probabilidad. Se agrega ademas un factor oculto de
     "habilidad/azar" que el modelo no ve, como en la vida real.
  4. El modelo se entrena SOLO con esos resultados 0/1, nunca con las
     probabilidades. Asi tiene que "descubrir" los patrones, igual que haria
     con datos reales. (Las probabilidades verdaderas se guardan aparte solo
     para medir el techo teorico de acierto en el reporte de entrenamiento.)

Esta verdad de terreno es INDEPENDIENTE de las formulas de ai_engine.py
(otras variables, otras formas, estocastica y con interacciones), para que
el modelo no se limite a copiar las formulas viejas.

Cuando existan datos reales, se reemplazan con load_csv() (ver docs/MODELO.md)
sin tocar nada mas: las caracteristicas salen del mismo features.py.

Este modulo usa numpy y SOLO se importa al entrenar, nunca en el servidor.
"""

import csv
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.ml import features as F

Point = Tuple[float, float]
HOOP = (50.0, 10.0)
FT = 0.5  # pies por unidad de cancha en baloncesto

# Tamanos por defecto: "posesiones" o "saques" sinteticos por deporte.
DEFAULT_SAMPLES = 80_000


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _interp(x: float, knots: Sequence[Tuple[float, float]]) -> float:
    """Interpolacion lineal entre puntos (x, y); fuera del rango usa el extremo."""
    if x <= knots[0][0]:
        return knots[0][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return knots[-1][1]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _seg_info(p: Point, a: Point, b: Point) -> Tuple[float, float]:
    """(distancia del punto al segmento a-b en unidades, posicion t en 0..1 de la proyeccion)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = dx * dx + dy * dy
    if n < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1]), 0.0
    t = _clamp(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / n, 0.0, 1.0)
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy)), t


# ============================================================================
# VERDAD DE TERRENO: BALONCESTO
# ============================================================================
# Porcentaje de acierto segun distancia al aro (pies). Aproximacion a lo que
# se ve en estadisticas publicas de la NBA/FIBA con tiros "promedio":
#   en el aro ~62-64%, media distancia ~40-45%, triples ~36%, y cae rapido
#   en tiros muy lejanos (mas de 30 pies).
_SHOT_2PT_KNOTS = [(0, 0.64), (5, 0.62), (9, 0.50), (14, 0.43), (19, 0.41), (23.75, 0.40)]
_SHOT_3PT_KNOTS = [(23.75, 0.365), (27, 0.36), (30, 0.33), (35, 0.24), (42, 0.14), (50, 0.08)]
# Cambio en la probabilidad (puntos de porcentaje /100) segun a cuantos pies
# esta el defensor mas cercano. Tracking publico: "muy marcado" (0-2 pies)
# pierde ~10-14 puntos; "marcado" (2-4) ~6; "abierto" (4-6) ~2; "libre" (6+)
# gana un poco.
_SHOT_CONTEST_KNOTS = [(0, -0.14), (2, -0.11), (4, -0.065), (6, -0.02), (8, 0.005), (12, 0.015)]
_CONTEST_RIM_MULT = 1.25        # cerca del aro un defensor encima molesta mas
_CONTEST_THREE_MULT = 0.85      # en triples molesta algo menos
_LINE_BLOCK_PENALTY = 0.05      # defensor justo en la linea tirador-aro (mano en la cara)
_RIM_PROTECTOR_PENALTY = 0.035  # por cada defensor protegiendo el aro (max 2) en tiros cerca del aro
_CROWD_PENALTY = 0.015          # por cada defensor extra a menos de 6 pies (doble marca)
_SHOOTER_SKILL_SIGMA = 0.30     # habilidad/azar oculto (desviacion en escala logit)


def shot_truth(holder: Point, defenders: Sequence[Point], rng: random.Random) -> Tuple[float, float]:
    """(prob. observable, prob. real con habilidad oculta) de que el tirador anote."""
    d_ft = math.hypot(holder[0] - HOOP[0], holder[1] - HOOP[1]) * FT
    base = _interp(d_ft, _SHOT_3PT_KNOTS if d_ft >= 23.75 else _SHOT_2PT_KNOTS)
    dists = sorted(math.hypot(holder[0] - d[0], holder[1] - d[1]) * FT for d in defenders)
    nearest = dists[0] if dists else 12.0
    mult = _CONTEST_RIM_MULT if d_ft < 8 else (_CONTEST_THREE_MULT if d_ft >= 23.75 else 1.0)
    p = base + mult * _interp(nearest, _SHOT_CONTEST_KNOTS)
    lane = min((_seg_info(d, holder, HOOP)[0] * FT for d in defenders), default=8.0)
    p -= _LINE_BLOCK_PENALTY * max(0.0, 1.0 - lane / 4.0)
    if d_ft < 10:
        protectors = sum(1 for d in defenders if math.hypot(d[0] - HOOP[0], d[1] - HOOP[1]) * FT <= 8.0)
        p -= _RIM_PROTECTOR_PENALTY * min(2, protectors)
    p -= _CROWD_PENALTY * max(0, sum(1 for x in dists if x < 6.0) - 1)
    p_obs = _clamp(p, 0.03, 0.85)
    p_real = _sigmoid(_logit(p_obs) + rng.gauss(0.0, _SHOOTER_SKILL_SIGMA))
    return p_obs, p_real


# Pase: probabilidad de COMPLETARLO (llega a las manos del receptor).
_PASS_BASE = 0.97               # un pase sin oposicion se completa ~97%
_INTERCEPT_PEAK = 0.55          # prob. de que un defensor pegado a la linea la corte
_INTERCEPT_WIDTH_FT = 2.4       # que tan ancho es el "radio de accion" de un defensor en el carril
_LONG_PASS_START_FT = 25.0      # desde aqui los pases largos pierden precision
_LONG_PASS_LOSS_PER_FT = 0.004
_RECEIVER_GUARDED_LOSS = 0.10   # perdida si el receptor tiene al defensor encima
_PASSER_PRESSED_LOSS = 0.06     # perdida si el pasador esta muy presionado
_PASSER_SKILL_SIGMA = 0.35


def pass_truth(holder: Point, receiver: Point, defenders: Sequence[Point],
               rng: random.Random) -> Tuple[float, float]:
    length_ft = math.hypot(holder[0] - receiver[0], holder[1] - receiver[1]) * FT
    p_safe = 1.0
    r_free = 12.0
    h_free = 12.0
    for d in defenders:
        dist_u, t = _seg_info(d, holder, receiver)
        dist_ft = dist_u * FT
        # Un pase largo da mas tiempo a interceptarlo; el centro del carril es el punto mas vulnerable.
        time_factor = 0.6 + length_ft / 40.0
        position_factor = 0.6 + 0.4 * math.sin(math.pi * t)
        hazard = _INTERCEPT_PEAK * math.exp(-((dist_ft / _INTERCEPT_WIDTH_FT) ** 2)) * time_factor * position_factor
        p_safe *= 1.0 - min(0.85, hazard)
        r_free = min(r_free, math.hypot(receiver[0] - d[0], receiver[1] - d[1]) * FT)
        h_free = min(h_free, math.hypot(holder[0] - d[0], holder[1] - d[1]) * FT)
    p = _PASS_BASE * p_safe
    p *= 1.0 - _LONG_PASS_LOSS_PER_FT * max(0.0, length_ft - _LONG_PASS_START_FT)
    p *= 1.0 - _RECEIVER_GUARDED_LOSS * max(0.0, 1.0 - r_free / 3.5)
    p *= 1.0 - _PASSER_PRESSED_LOSS * max(0.0, 1.0 - h_free / 3.0)
    p_obs = _clamp(p, 0.05, 0.985)
    return p_obs, _sigmoid(_logit(p_obs) + rng.gauss(0.0, _PASSER_SKILL_SIGMA))


# Penetracion: "exito" = termina en canasta o en falta a favor.
_DRIVE_BEAT_FLOOR = 0.30        # prob. minima de superar al defensor directo (aun pegado, a veces se logra)
_DRIVE_BEAT_RANGE = 0.62        # cuanto sube si el defensor esta lejos
_DRIVE_BEAT_MID_FT = 3.2        # distancia a la que se supera al defensor ~ mitad de las veces
_DRIVE_LANE_PENALTY = 0.25      # un defensor justo en el camino al aro reduce hasta 25%
_DRIVE_FINISH_BASE = 0.60       # acierto de una bandeja sin ayuda defensiva
_DRIVE_HELP_PENALTY = 0.075     # por cada defensor que ayuda en la pintura (max 3)
_DRIVE_FOUL_BONUS = 0.12        # fraccion de los fallos que se convierten en falta a favor
_DRIVE_TURNOVER_PER_FT = 0.003  # riesgo de perdida por pie recorrido en la penetracion
_DRIVE_SKILL_SIGMA = 0.35


def drive_truth(holder: Point, defenders: Sequence[Point], rng: random.Random) -> Tuple[float, float]:
    d_ft = math.hypot(holder[0] - HOOP[0], holder[1] - HOOP[1]) * FT
    on_ball = min((math.hypot(holder[0] - d[0], holder[1] - d[1]) * FT for d in defenders), default=12.0)
    lane = min((_seg_info(d, holder, HOOP)[0] * FT for d in defenders), default=8.0)
    helpers = sum(1 for d in defenders if math.hypot(d[0] - HOOP[0], d[1] - HOOP[1]) * FT <= 9.0)
    beat = _DRIVE_BEAT_FLOOR + _DRIVE_BEAT_RANGE * _sigmoid((on_ball - _DRIVE_BEAT_MID_FT) / 1.4)
    beat *= (1.0 - _DRIVE_LANE_PENALTY) + _DRIVE_LANE_PENALTY * min(1.0, lane / 4.0)
    finish = _DRIVE_FINISH_BASE - _DRIVE_HELP_PENALTY * min(3, helpers)
    finish_or_foul = finish + _DRIVE_FOUL_BONUS * (1.0 - finish)
    p = beat * finish_or_foul * (1.0 - _DRIVE_TURNOVER_PER_FT * d_ft)
    p_obs = _clamp(p, 0.04, 0.9)
    return p_obs, _sigmoid(_logit(p_obs) + rng.gauss(0.0, _DRIVE_SKILL_SIGMA))


# ============================================================================
# VERDAD DE TERRENO: TENIS (el sacador gana el punto)
# ============================================================================
_SERVE_SPEED_MPH = (95.0, 128.0)   # rango de velocidad de un primer saque profesional/amateur avanzado
_MPH_TO_FTS = 1.467                # millas/hora -> pies/segundo
_FLIGHT_AVG_SPEED_RATIO = 0.75     # la pelota pierde velocidad en el aire: promedio ~75% de la inicial
_POST_BOUNCE_SPEED_RATIO = 0.60    # tras el bote conserva ~60% de la velocidad
_RETURNER_REACTION_S = 0.22        # tiempo de reaccion visual (~0.2 s)
_RETURNER_HEAD_START_S = 0.25      # lee el saque (lanzamiento, postura) y se adelanta ~0.25 s
_RETURNER_SPEED_FTS = 14.0         # velocidad lateral media de desplazamiento (con aceleracion)
_RACKET_REACH_FT = 5.5             # brazo + raqueta: no hace falta llegar al punto exacto
_KICK_FT = 2.5                     # saques abiertos "patean" hacia afuera y desvian hasta ~2.5 pies mas
_UNRETURNABLE_FLOOR = 0.04         # incluso saques faciles a veces son ace/error forzado
_UNRETURNABLE_RANGE = 0.80         # prob. extra de no poder devolver cuando esta fuera de alcance
_RALLY_BASE = 0.50                 # si devuelve normal, el punto es ~50/50 (el sacador tiene ligera ventaja con saque profundo)
_RALLY_STRETCH_BONUS = 0.14        # devolver estirado favorece al sacador
_RALLY_DEEP_BONUS = 0.05           # saque profundo = devolucion mas debil
_FIRST_IN_BASE = 0.72              # primer saque dentro: ~60-70% en el tenis profesional si se apunta bien
_AIM_RISK_LOSS = 0.22              # apuntar a menos de ~1.3 pies de una linea resta hasta 22 puntos
_SPEED_IN_LOSS_PER_MPH = 0.004     # mas velocidad = menos saques dentro
_OUT_OF_BOX_IN = 0.03              # apuntar fuera de la caja = casi falta segura
_SECOND_IN = 0.93                  # segundo saque dentro
_SECOND_WIN = 0.46                 # el que saca gana ~45-50% con segundo saque
_SERVE_NOISE_SIGMA = 0.25


def serve_truth(server: Point, returner: Point, target: Point, rng: random.Random) -> Tuple[float, float]:
    sx, sy = server
    rx, ry = returner
    tx, ty = target
    xf, yf = F.TENNIS_FT_PER_X, F.TENNIS_FT_PER_Y
    speed = rng.uniform(*_SERVE_SPEED_MPH)
    v = speed * _MPH_TO_FTS

    length = math.hypot((tx - sx) * xf, (ty - sy) * yf)
    t_flight = length / (_FLIGHT_AVG_SPEED_RATIO * v)
    post_dist = abs(ry - ty) * yf
    t_post = post_dist / (_POST_BOUNCE_SPEED_RATIO * v)

    # Hacia donde viaja la pelota despues del bote (misma recta sacador -> bote).
    dy = ty - sy
    arrival_x = tx if abs(dy) < 1e-9 else tx + (tx - sx) * (ry - ty) / dy
    arrival_x = _clamp(arrival_x, -50.0, 150.0)
    wide = (13.5 - min(13.5, max(0.0, min(tx, 100 - tx)) * xf)) / 13.5
    miss = abs(arrival_x - rx) * xf + _KICK_FT * wide

    move_time = max(0.0, _RETURNER_HEAD_START_S + 0.5 * t_flight + t_post - _RETURNER_REACTION_S)
    reachable = _RACKET_REACH_FT + _RETURNER_SPEED_FTS * move_time
    p_unreturnable = _UNRETURNABLE_FLOOR + _UNRETURNABLE_RANGE * _sigmoid((miss - reachable) / 1.6)

    stretch = min(1.0, miss / reachable)
    depth_ft = (ty - F.TENNIS_SERVICE_LINE_Y) * yf
    deep = 1.0 - _clamp(depth_ft, 0.0, 16.7) / 16.7
    p_rally = _RALLY_BASE + _RALLY_STRETCH_BONUS * stretch + _RALLY_DEEP_BONUS * deep
    p_first_won = p_unreturnable + (1.0 - p_unreturnable) * p_rally

    side = min(13.5, max(0.0, min(tx, 100 - tx)) * xf)
    center = min(13.5, abs(tx - 50.0) * xf)
    aim_risk = max(math.exp(-side / 1.3), math.exp(-center / 1.3), math.exp(-max(depth_ft, 0.0) / 1.3))
    if ty < F.TENNIS_SERVICE_LINE_Y or ty > F.TENNIS_NET_Y:
        p_in = _OUT_OF_BOX_IN
    else:
        p_in = _clamp(_FIRST_IN_BASE - _AIM_RISK_LOSS * aim_risk - _SPEED_IN_LOSS_PER_MPH * (speed - 110.0), 0.05, 0.95)

    p_second = _SECOND_IN * (_SECOND_WIN + 0.08 * stretch)
    p = p_in * p_first_won + (1.0 - p_in) * p_second
    p_obs = _clamp(p, 0.03, 0.97)
    return p_obs, _sigmoid(_logit(p_obs) + rng.gauss(0.0, _SERVE_NOISE_SIGMA))


# ============================================================================
# GENERADORES DE SITUACIONES
# ============================================================================
def _clip_pt(x: float, y: float) -> Point:
    return (_clamp(x, 0.0, 100.0), _clamp(y, 0.0, 100.0))


def _sample_basketball_state(rng: random.Random) -> Tuple[List[Point], List[Point], Point]:
    """Una situacion de media cancha 5v5: atacantes, defensores y balon."""
    attackers: List[Point] = []
    for _ in range(5):
        if rng.random() < 0.7:
            r = rng.uniform(3.0, 58.0)
            th = rng.uniform(0.03 * math.pi, 0.97 * math.pi)
            attackers.append(_clip_pt(HOOP[0] + r * math.cos(th), HOOP[1] + r * math.sin(th)))
        else:
            attackers.append((rng.uniform(5, 95), rng.uniform(8, 90)))
    holder = rng.choice(attackers)
    ball = _clip_pt(holder[0] + rng.gauss(0, 0.8), holder[1] + rng.gauss(0, 0.8))

    # Estilo de marcacion de la situacion: pegada, normal o suelta (en unidades; 1 u = 0.5 pies).
    style = rng.choices(["tight", "normal", "loose"], weights=[0.35, 0.40, 0.25])[0]
    lo, hi = {"tight": (1.5, 6.0), "normal": (5.0, 16.0), "loose": (14.0, 40.0)}[style]
    marks = attackers[:]
    rng.shuffle(marks)
    defenders: List[Point] = []
    for mark in marks:
        if rng.random() < 0.10:                       # defensor "perdido": en cualquier lugar
            defenders.append((rng.uniform(0, 100), rng.uniform(0, 95)))
            continue
        off = rng.uniform(lo, hi)
        to_hoop = math.atan2(HOOP[1] - mark[1], HOOP[0] - mark[0])
        ang = to_hoop + rng.gauss(0, 0.6) if rng.random() < 0.7 else rng.uniform(-math.pi, math.pi)
        defenders.append(_clip_pt(mark[0] + off * math.cos(ang), mark[1] + off * math.sin(ang)))
    if rng.random() < 0.30:                           # ayudas en la pintura
        for i in rng.sample(range(5), rng.choice([1, 2])):
            defenders[i] = (rng.uniform(40, 60), rng.uniform(10, 26))
    return attackers, defenders, ball


def generate_basketball(n_states: int = DEFAULT_SAMPLES, seed: int = 42) -> Dict[str, Dict[str, np.ndarray]]:
    """Genera n_states situaciones y, de cada una, un intento por cabeza:
    un tiro del portador, un pase a un companero al azar y una penetracion.
    Devuelve {cabeza: {"X", "y", "p_obs"}}; y es el resultado 0/1 sorteado."""
    rng = random.Random(seed)
    rows: Dict[str, List[List[float]]] = {"shot": [], "pass": [], "drive": []}
    ys: Dict[str, List[int]] = {"shot": [], "pass": [], "drive": []}
    ps: Dict[str, List[float]] = {"shot": [], "pass": [], "drive": []}
    for _ in range(n_states):
        attackers, defenders, ball = _sample_basketball_state(rng)
        h = F.basketball_holder_index(attackers, ball)
        holder = attackers[h]
        others = [a for i, a in enumerate(attackers) if i != h]
        receiver = rng.choice(others)

        for head, truth, extra in (
            ("shot", shot_truth, None),
            ("pass", pass_truth, receiver),
            ("drive", drive_truth, None),
        ):
            if head == "pass":
                p_obs, p_real = truth(holder, receiver, defenders, rng)
            else:
                p_obs, p_real = truth(holder, defenders, rng)
            rows[head].append(F.basketball_features(head, holder, defenders, extra))
            ys[head].append(1 if rng.random() < p_real else 0)
            ps[head].append(p_obs)
    return {
        head: {"X": np.asarray(rows[head], dtype=float), "y": np.asarray(ys[head], dtype=float),
               "p_obs": np.asarray(ps[head], dtype=float)}
        for head in rows
    }


def _sample_tennis_state(rng: random.Random) -> Tuple[Point, Point, Point]:
    server = (rng.uniform(20, 80), rng.uniform(86, 98))
    returner = (rng.uniform(12, 88), rng.uniform(1, 24))
    u = rng.random()
    if u < 0.08:                                       # saques fuera de la caja (largos o en la red)
        ty = rng.choice([rng.uniform(14, 28.2), rng.uniform(50.5, 58)])
        tx = rng.uniform(0, 100)
    else:
        ty = rng.uniform(F.TENNIS_SERVICE_LINE_Y, F.TENNIS_NET_Y)
        if u < 0.33:                                   # apuntando cerca de las lineas (riesgoso)
            side = rng.choice(["lat", "cen", "srv"])
            if side == "lat":
                tx = rng.choice([rng.uniform(0, 6), rng.uniform(94, 100)])
            elif side == "cen":
                tx = 50 + rng.choice([-1, 1]) * rng.uniform(0, 6)
            else:
                tx, ty = rng.uniform(0, 100), rng.uniform(F.TENNIS_SERVICE_LINE_Y, F.TENNIS_SERVICE_LINE_Y + 3)
        else:
            tx = rng.uniform(0, 100)
    return server, returner, (tx, ty)


def generate_tennis(n_states: int = DEFAULT_SAMPLES, seed: int = 42) -> Dict[str, Dict[str, np.ndarray]]:
    rng = random.Random(seed + 1)
    xs, ys, ps = [], [], []
    for _ in range(n_states):
        server, returner, target = _sample_tennis_state(rng)
        p_obs, p_real = serve_truth(server, returner, target, rng)
        xs.append(F.tennis_serve_features(server, returner, target))
        ys.append(1 if rng.random() < p_real else 0)
        ps.append(p_obs)
    return {"serve": {"X": np.asarray(xs, dtype=float), "y": np.asarray(ys, dtype=float),
                      "p_obs": np.asarray(ps, dtype=float)}}


# ============================================================================
# DATOS REALES (cuando existan): lectura desde CSV
# ============================================================================
def _pt(row: Dict[str, str], prefix: str) -> Optional[Point]:
    x, y = row.get(f"{prefix}_x", "").strip(), row.get(f"{prefix}_y", "").strip()
    return (float(x), float(y)) if x != "" and y != "" else None


def load_csv(path: str, sport: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Lee datos REALES con el esquema descrito en docs/MODELO.md y devuelve el
    mismo formato que los generadores sinteticos (sin p_obs).

    Baloncesto: columnas head (shot|pass|drive), outcome (0/1), ball_x, ball_y,
      a1_x,a1_y .. a5_x,a5_y (atacantes), d1_x,d1_y .. d5_x,d5_y (defensores EN
      EL MOMENTO de la accion) y receiver (1-5, solo para head=pass).
    Tenis: outcome (1 = el sacador gano el punto), server_x, server_y,
      returner_x, returner_y, target_x, target_y.
    """
    rows: Dict[str, List[List[float]]] = {}
    ys: Dict[str, List[int]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            outcome = int(float(row["outcome"]))
            if outcome not in (0, 1):
                raise ValueError(f"outcome debe ser 0 o 1, llego {outcome}")
            if sport == "tennis":
                feats = F.tennis_serve_features(_pt(row, "server"), _pt(row, "returner"), _pt(row, "target"))
                head = "serve"
            else:
                head = row["head"].strip()
                attackers = [p for p in (_pt(row, f"a{i}") for i in range(1, 6)) if p is not None]
                defenders = [p for p in (_pt(row, f"d{i}") for i in range(1, 6)) if p is not None]
                ball = _pt(row, "ball")
                holder_i = F.basketball_holder_index(attackers, ball)
                receiver = None
                if head == "pass":
                    receiver = _pt(row, f"a{int(row['receiver'])}")
                feats = F.basketball_features(head, attackers[holder_i], defenders, receiver)
            rows.setdefault(head, []).append(feats)
            ys.setdefault(head, []).append(outcome)
    return {h: {"X": np.asarray(rows[h], dtype=float), "y": np.asarray(ys[h], dtype=float)} for h in rows}
