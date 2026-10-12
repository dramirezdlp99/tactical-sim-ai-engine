"""
Guardar en: tactical-sim-ai-engine/app/services/defense_response.py

"CUERPO TECNICO RIVAL": PREDICE COMO REACCIONA LA DEFENSA.

La especificacion pide que la IA simule la respuesta defensiva del equipo
contrario con base en patrones de marcacion y coberturas espaciales. Antes,
los defensores se quedaban quietos donde el entrenador los dibujo. Ahora, dada
la jugada propuesta, este modulo estima DONDE ESTARIAN los defensores un
instante despues de reaccionar, y el modelo evalua la jugada contra esa
defensa ya reajustada.

Es un modelo de reglas (no entrenado), pero parametrizado con conceptos
reales de defensa. Baloncesto, en 4 pasos:

  1. EMPAREJAMIENTO HOMBRE A HOMBRE (asignacion voraz): se calculan las
     distancias entre cada defensor y cada atacante, se ordenan de menor a
     mayor y se va asignando la pareja mas cercana disponible. Coste O(n^2 log n)
     con n=5 (25 parejas); NO se prueban permutaciones porque esto se ejecuta
     4000 veces por peticion en Monte Carlo.
  2. POSICION "DEL LADO DEL ARO": cada defensor quiere pararse entre su marca
     y el aro, a una distancia corta de ella (mas pegado si su marca tiene el
     balon).
  3. AYUDA EN LA PINTURA: si el balon esta cerca del aro, los defensores cuya
     marca esta mas lejos del balon (lado debil) se cierran hacia la pintura
     para ayudar, a costa de soltar un poco a su marca.
  4. PASO MAXIMO: nadie se teletransporta; en una reaccion cada defensor se
     mueve como maximo MAX_REACTION_STEP unidades.

Tenis: el restador anticipa el saque y se desplaza hacia el punto de bote,
limitado por su alcance de reaccion.

Todas las posiciones devueltas quedan dentro de 0-100.
"""

import math
from typing import List, NamedTuple, Tuple

from app.services.geometry import BASKETBALL_HOOP

Point = Tuple[float, float]

# ---------------------------------------------------------------------------
# Constantes de baloncesto (unidades 0-100; 1 unidad ~ 0.5 pies)
# ---------------------------------------------------------------------------
# Paso maximo por reaccion. 10 unidades ~ 5 pies: lo que un defensor recorre
# con un deslizamiento defensivo en unos 0.6 s. Es el limite mas importante:
# evita que la IA "teletransporte" defensores y subestime el tiro libre.
MAX_REACTION_STEP = 10.0
# Distancia a la que el defensor se para de su marca, del lado del aro.
# 3 unidades ~ 1.5 pies (marca pegada al balon); 5 ~ 2.5 pies (sin balon,
# deja un poco mas de espacio para poder ayudar).
GOAL_SIDE_OFFSET_ON_BALL = 3.0
GOAL_SIDE_OFFSET_OFF_BALL = 5.0
# Si el balon esta a menos de esta distancia del aro, se activa la ayuda en la
# pintura (30 unidades ~ 15 pies: dentro del area de media distancia corta).
HELP_TRIGGER_DIST = 30.0
# Cuantos defensores del lado debil se cierran a ayudar.
HELP_DEFENDERS = 2
# Que tanto se desplaza el ayudante desde su marca hacia la pintura (0 = no
# ayuda, 1 = abandona del todo a su marca). 0.5 = se queda a medio camino.
HELP_BLEND = 0.5
# Punto de ayuda: sobre la linea aro-balon, a esta distancia del aro.
HELP_POINT_FROM_HOOP = 10.0

SCHEME_MAN = "Hombre a hombre"
SCHEME_MAN_HELP = "Hombre a hombre con ayuda en la pintura"

# ---------------------------------------------------------------------------
# Constantes de tenis
# ---------------------------------------------------------------------------
# Pies reales por unidad (los mismos que app/ml/features.py: ejes distintos).
_TENNIS_FT_X = 27.0 / 100.0
_TENNIS_FT_Y = 78.0 / 100.0
# Paso maximo de anticipacion: ~0.25 s de ventaja leyendo el saque a ~15 pies/s
# = ~4 pies. Es lo que el restador puede "adelantar" antes de ver el bote.
TENNIS_MAX_ANTICIPATION_FT = 4.0
# Fraccion de la distancia al punto de bote que intenta cubrir anticipando.
TENNIS_ANTICIPATION_FRACTION = 0.5
# El restador prioriza moverse de lado y casi no adelanta: no abandona su
# posicion de fondo. 0.25 = la distancia hacia adelante cuenta 4 veces menos.
TENNIS_FORWARD_WEIGHT = 0.25

SCHEME_RETURN_ANTICIPATE = "Restador anticipa hacia el punto de bote"
SCHEME_RETURN_WAIT = "Restador en posicion de espera"


class DefenseReaction(NamedTuple):
    """Resultado: posiciones predichas [(player_id, (x, y))] en el MISMO orden que la entrada, y el esquema."""
    positions: List[Tuple[str, Point]]
    scheme: str


def _clamp100(v: float) -> float:
    return 0.0 if v < 0.0 else (100.0 if v > 100.0 else v)


def _move_towards(cur: Point, target: Point, max_step: float) -> Point:
    dx, dy = target[0] - cur[0], target[1] - cur[1]
    dist = math.hypot(dx, dy)
    if dist > max_step and dist > 0.0:
        k = max_step / dist
        dx, dy = dx * k, dy * k
    return (_clamp100(cur[0] + dx), _clamp100(cur[1] + dy))


def greedy_match(defenders: List[Point], attackers: List[Point]) -> List[int]:
    """Asignacion voraz defensor -> atacante. Devuelve, para cada defensor, el
    indice de su marca (o -1 si sobran defensores). O(n*m*log(n*m))."""
    pairs = sorted(
        (math.hypot(d[0] - a[0], d[1] - a[1]), di, ai)
        for di, d in enumerate(defenders)
        for ai, a in enumerate(attackers)
    )
    mark = [-1] * len(defenders)
    taken = [False] * len(attackers)
    left = min(len(defenders), len(attackers))
    for _, di, ai in pairs:
        if mark[di] == -1 and not taken[ai]:
            mark[di] = ai
            taken[ai] = True
            left -= 1
            if left == 0:
                break
    return mark


def predict_basketball_defense(
    attackers: List[Tuple[str, Point]],
    defenders: List[Tuple[str, Point]],
    ball: Point,
    holder_idx: int,
    hoop: Point = BASKETBALL_HOOP,
) -> DefenseReaction:
    """Posiciones de los defensores despues de reaccionar a la jugada.

    attackers / defenders: listas de (player_id, (x, y)). holder_idx: indice
    del atacante con el balon (el mismo que usa el resto del motor).
    """
    att_xy = [a[1] for a in attackers]
    def_xy = [d[1] for d in defenders]
    marks = greedy_match(def_xy, att_xy)
    hx, hy = hoop

    ball_to_hoop = math.hypot(ball[0] - hx, ball[1] - hy)
    help_active = ball_to_hoop <= HELP_TRIGGER_DIST

    # Defensores que ayudan: los off-ball cuya marca esta MAS LEJOS del balon (lado debil).
    helpers = set()
    if help_active:
        off_ball = [(math.hypot(att_xy[m][0] - ball[0], att_xy[m][1] - ball[1]), di)
                    for di, m in enumerate(marks) if m != -1 and m != holder_idx]
        off_ball.sort(reverse=True)
        helpers = {di for _, di in off_ball[:HELP_DEFENDERS]}
        # Punto de ayuda: entre el aro y el balon, dentro de la pintura.
        if ball_to_hoop > 1e-9:
            ux, uy = (ball[0] - hx) / ball_to_hoop, (ball[1] - hy) / ball_to_hoop
        else:
            ux, uy = 0.0, 1.0
        help_pt = (hx + ux * HELP_POINT_FROM_HOOP, hy + uy * HELP_POINT_FROM_HOOP)

    out: List[Tuple[str, Point]] = []
    unmatched = False
    for di, (pid, cur) in enumerate(defenders):
        m = marks[di]
        if m == -1:
            # Defensor sobrante (menos atacantes que defensores): protege la pintura.
            unmatched = True
            tgt = (hx, hy + HELP_POINT_FROM_HOOP)
        else:
            mx, my = att_xy[m]
            dist_hoop = math.hypot(hx - mx, hy - my)
            offset = GOAL_SIDE_OFFSET_ON_BALL if m == holder_idx else GOAL_SIDE_OFFSET_OFF_BALL
            if dist_hoop > 1e-9:
                k = min(offset, dist_hoop) / dist_hoop
                tgt = (mx + (hx - mx) * k, my + (hy - my) * k)
            else:
                tgt = (mx, my)
            if di in helpers:
                tgt = (tgt[0] + HELP_BLEND * (help_pt[0] - tgt[0]), tgt[1] + HELP_BLEND * (help_pt[1] - tgt[1]))
        out.append((pid, _move_towards(cur, tgt, MAX_REACTION_STEP)))

    scheme = SCHEME_MAN_HELP if helpers else SCHEME_MAN
    if unmatched:
        scheme += " y defensor libre en la pintura"
    return DefenseReaction(out, scheme)


def predict_tennis_returner(server: Point, returner: Tuple[str, Point], target: Point) -> DefenseReaction:
    """Posicion anticipada del restador: se desplaza hacia el punto de bote,
    limitado por TENNIS_MAX_ANTICIPATION_FT y priorizando el movimiento lateral."""
    pid, (rx, ry) = returner
    gx = (target[0] - rx) * _TENNIS_FT_X
    gy = (target[1] - ry) * _TENNIS_FT_Y * TENNIS_FORWARD_WEIGHT
    dist = math.hypot(gx, gy)
    step = min(TENNIS_MAX_ANTICIPATION_FT, TENNIS_ANTICIPATION_FRACTION * dist)
    if dist < 1e-9 or step < 1e-6:
        return DefenseReaction([(pid, (_clamp100(rx), _clamp100(ry)))], SCHEME_RETURN_WAIT)
    k = step / dist
    nx = _clamp100(rx + (target[0] - rx) * k)
    ny = _clamp100(ry + (target[1] - ry) * TENNIS_FORWARD_WEIGHT * k)
    return DefenseReaction([(pid, (nx, ny))], SCHEME_RETURN_ANTICIPATE)
