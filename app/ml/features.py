"""
Guardar en: tactical-sim-ai-engine/app/ml/features.py

EXTRACCION DE CARACTERISTICAS ("features") -- UNA SOLA FUENTE DE VERDAD.

Que es una "caracteristica": un numero que resume la situacion para que el
modelo pueda razonar (por ejemplo: "a cuantos pies esta el defensor mas
cercano al tirador"). El modelo nunca ve las coordenadas crudas, solo estas
caracteristicas.

Por que este archivo es tan importante: el MISMO codigo lo usan
  1) el entrenamiento (app/ml/dataset.py + app/ml/train.py) y
  2) la inferencia en produccion (app/services/ai_engine.py).
Si cada lado calculara las caracteristicas por separado, cualquier pequena
diferencia (otra unidad, otro recorte) haria que el modelo reciba en
produccion numeros distintos a los que vio al entrenar. A eso se le llama
"training-serving skew" y arruina las predicciones sin dar ningun error.
Aqui es imposible: hay una sola implementacion.

Reglas de este modulo:
  - Python puro (modulo math). NO usa numpy, porque corre en el servidor
    gratuito de Render (512 MB de RAM).
  - Las posiciones entran como tuplas (x, y) en el sistema 0-100 del
    frontend. Aqui se convierten a PIES reales para que las caracteristicas
    tengan sentido fisico y sean faciles de explicar.
  - Si se cambia cualquier formula de este archivo, hay que REENTRENAR
    (python -m app.ml.train) y subir la FEATURE_SCHEMA_VERSION.
"""

import math
from typing import Dict, List, Sequence, Tuple

from app.services.geometry import BASKETBALL_HOOP, _point_to_segment_distance

Point = Tuple[float, float]

# Version del "esquema" de caracteristicas. Se guarda dentro del JSON del
# modelo; si no coincide al cargarlo, el modelo se ignora (fallback seguro).
FEATURE_SCHEMA_VERSION = 1

# ----------------------------------------------------------------------------
# Conversion de coordenadas (0-100) a pies
# ----------------------------------------------------------------------------
# Baloncesto: media cancha FIBA/NBA ~ 50 pies de ancho x 47 de largo. Con
# 0-100 en ambos ejes, 1 unidad ~ 0.5 pies. Se usa 0.5 en los dos ejes
# (la diferencia 0.47 vs 0.5 en el eje largo es despreciable).
BASKET_FT_PER_UNIT = 0.5
# Distancia de la linea de triple al aro (NBA, tramo curvo): 23.75 pies.
THREE_POINT_FT = 23.75

# Tenis (cancha de individuales vista en vertical): 27 pies de ancho en el
# eje X (0-100) y 78 pies de largo en el eje Y (0-100). SUPUESTO: x=0 y x=100
# son las lineas laterales de individuales e y=0 / y=100 las lineas de fondo.
# Por eso 1 unidad de X vale 0.27 pies y 1 unidad de Y vale 0.78 pies: los
# ejes NO son iguales y hay que convertirlos por separado.
TENNIS_FT_PER_X = 27.0 / 100.0
TENNIS_FT_PER_Y = 78.0 / 100.0
# Caja de saque del restador (segun el frontend): entre la linea de saque
# (y ~ 28.6) y la red (y = 50).
TENNIS_SERVICE_LINE_Y = 28.6
TENNIS_NET_Y = 50.0

# Recortes ("caps"): mas alla de estas distancias, estar mas lejos ya no
# ayuda. Recortar hace que el modelo lineal se parezca mas a la realidad.
OPEN_CAP_FT = 10.0       # un defensor a 10+ pies = tirador completamente solo
LANE_CAP_FT = 8.0        # un defensor a 8+ pies de la linea = linea libre
NEAR_RADIUS_FT = 8.0     # radio para contar defensores "encima"
LANE_NEAR_FT = 4.0       # un defensor a <= 4 pies de la linea de pase la molesta
PAINT_RADIUS_FT = 10.0   # defensores que protegen el aro
RIM_RANGE_FT = 12.0      # de 0 a 12 pies el efecto "cerca del aro" se desvanece

# ----------------------------------------------------------------------------
# Nombres de caracteristicas por cabeza (el ORDEN importa: es el de los pesos)
# ----------------------------------------------------------------------------
SHOT_FEATURES: List[str] = [
    "dist_aro_ft",          # distancia del tirador al aro, en pies
    "dist_aro_ft_cuad",     # esa distancia al cuadrado / 100 (curva: cerca y lejos no son lineales)
    "es_triple",            # 1 si esta detras de la linea de 3
    "defensor_libre_ft",    # distancia al defensor mas cercano (recortada a 10 pies)
    "linea_tiro_libre_ft",  # distancia del defensor mas cercano a la linea tirador-aro (recortada)
    "defensores_encima",    # cuantos defensores a <= 8 pies del tirador
    "defensores_pintura",   # cuantos defensores a <= 10 pies del aro (proteccion de aro)
    "libre_x_cerca_aro",    # interaccion: defensor libre * cercania al aro
    "pintura_x_cerca_aro",  # interaccion: proteccion de aro * cercania al aro
    "libre_x_triple",       # interaccion: defensor libre * es_triple
    "libre_bajo_4",         # "bisagra": pies que le faltan al defensor para estar a 4 pies (marca pegada)
    "libre_bajo_6",         # "bisagra": idem para 6 pies
    "linea_bajo_4",         # "bisagra": defensor dentro de 4 pies de la linea tirador-aro
    "pies_sobre_triple",    # pies de tiro por detras de la linea de 3 (tiros muy lejanos)
    "tiro_cerca_aro",       # 1 si el tiro es a menos de 8 pies del aro
]

PASS_FEATURES: List[str] = [
    "largo_pase_ft",        # distancia pasador-receptor
    "carril_libre_ft",      # defensor mas cercano al segmento del pase (recortada a 8)
    "defensores_carril",    # cuantos defensores a <= 4 pies del segmento del pase
    "receptor_libre_ft",    # defensor mas cercano al receptor (recortada a 10)
    "pasador_libre_ft",     # defensor mas cercano al pasador (recortada a 10)
    "bloqueo_x_largo",      # interaccion: (8 - carril_libre) * largo / 10
    "pase_largo_extra",     # pies de pase por encima de 25
    "carril_cerrado",       # "bisagra": pies que le faltan al carril para tener 4 pies libres
    "carril_cerrado_cuad",  # idem al cuadrado (un defensor justo encima corta el pase mucho mas)
    "receptor_marcado",     # "bisagra": receptor con defensor a menos de 3.5 pies
    "pasador_presionado",   # "bisagra": pasador con defensor a menos de 3 pies
    "cerrado_x_largo",      # interaccion: carril_cerrado * largo / 10
]

DRIVE_FEATURES: List[str] = [
    "dist_aro_ft",          # distancia al aro
    "dist_aro_ft_cuad",     # idem al cuadrado / 100
    "defensor_balon_ft",    # defensor mas cercano al portador (recortada a 10)
    "camino_libre_ft",      # defensor mas cercano al camino portador-aro (recortada a 8)
    "ayudas_pintura",       # defensores a <= 10 pies del aro que pueden ayudar
    "camino_x_balon",       # interaccion: camino_libre * defensor_balon
    "dist_x_ayudas",        # interaccion: distancia * ayudas / 10
    "balon_bajo_4",         # "bisagra": defensor directo a menos de 4 pies
    "camino_bajo_4",        # "bisagra": defensor a menos de 4 pies del camino al aro
    "ayudas_max3",          # ayudas en la pintura, con tope de 3
]

SERVE_FEATURES: List[str] = [
    "fallo_lateral_ft",     # cuanto se desvia (en pies) la trayectoria del saque de donde esta el restador
    "fallo_lateral_cuad",   # idem al cuadrado / 10
    "carrera_bote_ft",      # distancia real del restador al punto de bote
    "largo_saque_ft",       # distancia sacador -> punto de bote
    "borde_lateral_ft",     # distancia del bote a la linea lateral (recortada a 13.5)
    "borde_central_ft",     # distancia del bote a la linea central de saque (recortada a 13.5)
    "prof_linea_saque_ft",  # pies entre el bote y la linea de saque (positivo = dentro; recortada)
    "fuera_de_caja",        # 1 si el bote cae fuera de la caja de saque (largo o en la red)
    "restador_fondo_ft",    # a cuantos pies de su linea de fondo esta el restador
    "fallo_x_corto",        # interaccion: fallo_lateral * (saque corto); un saque profundo da menos tiempo
    "fallo_x_ancho",        # interaccion: fallo_lateral * (saque abierto hacia la linea lateral)
    "fallo_sobre_7ft",      # "bisagra": pies de fallo lateral por encima de 7 (donde empieza a no llegar)
    "fallo_sobre_12ft",     # "bisagra": pies de fallo lateral por encima de 12 (casi seguro no llega)
]

FEATURE_NAMES: Dict[str, Dict[str, List[str]]] = {
    "basketball": {"shot": SHOT_FEATURES, "pass": PASS_FEATURES, "drive": DRIVE_FEATURES},
    "tennis": {"serve": SERVE_FEATURES},
}


# ----------------------------------------------------------------------------
# Baloncesto
# ----------------------------------------------------------------------------
def basketball_holder_index(attackers: Sequence[Point], ball: Point) -> int:
    """Indice del atacante que tiene el balon: el mas cercano a la posicion del balon."""
    best_i, best_d = 0, float("inf")
    for i, a in enumerate(attackers):
        d = math.hypot(a[0] - ball[0], a[1] - ball[1])
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def _ft_dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1]) * BASKET_FT_PER_UNIT


def _basketball_shot(holder: Point, defenders: Sequence[Point]) -> List[float]:
    d_ft = _ft_dist(holder, BASKETBALL_HOOP)
    nearest = OPEN_CAP_FT
    lane = LANE_CAP_FT
    encima = 0
    pintura = 0
    for d in defenders:
        dh = _ft_dist(holder, d)
        if dh < nearest:
            nearest = dh
        if dh <= NEAR_RADIUS_FT:
            encima += 1
        if _ft_dist(BASKETBALL_HOOP, d) <= PAINT_RADIUS_FT:
            pintura += 1
        dl = _point_to_segment_distance(d, holder, BASKETBALL_HOOP) * BASKET_FT_PER_UNIT
        if dl < lane:
            lane = dl
    rim_w = max(0.0, 1.0 - d_ft / RIM_RANGE_FT)
    is3 = 1.0 if d_ft >= THREE_POINT_FT else 0.0
    return [
        d_ft, d_ft * d_ft / 100.0, is3, nearest, lane, float(encima), float(pintura),
        nearest * rim_w, pintura * rim_w, nearest * is3,
        max(0.0, 4.0 - nearest), max(0.0, 6.0 - nearest), max(0.0, 4.0 - lane),
        max(0.0, d_ft - THREE_POINT_FT), 1.0 if d_ft < 8.0 else 0.0,
    ]


def _basketball_pass(holder: Point, receiver: Point, defenders: Sequence[Point]) -> List[float]:
    length = _ft_dist(holder, receiver)
    lane = LANE_CAP_FT
    in_lane = 0
    recv_free = OPEN_CAP_FT
    pass_free = OPEN_CAP_FT
    for d in defenders:
        dl = _point_to_segment_distance(d, holder, receiver) * BASKET_FT_PER_UNIT
        if dl < lane:
            lane = dl
        if dl <= LANE_NEAR_FT:
            in_lane += 1
        dr = _ft_dist(receiver, d)
        if dr < recv_free:
            recv_free = dr
        dh = _ft_dist(holder, d)
        if dh < pass_free:
            pass_free = dh
    return [
        length, lane, float(in_lane), recv_free, pass_free,
        (LANE_CAP_FT - lane) * length / 10.0, max(0.0, length - 25.0),
        max(0.0, 4.0 - lane), max(0.0, 4.0 - lane) ** 2, max(0.0, 3.5 - recv_free),
        max(0.0, 3.0 - pass_free), max(0.0, 4.0 - lane) * length / 10.0,
    ]


def _basketball_drive(holder: Point, defenders: Sequence[Point]) -> List[float]:
    d_ft = _ft_dist(holder, BASKETBALL_HOOP)
    on_ball = OPEN_CAP_FT
    lane = LANE_CAP_FT
    helpers = 0
    for d in defenders:
        dh = _ft_dist(holder, d)
        if dh < on_ball:
            on_ball = dh
        if _ft_dist(BASKETBALL_HOOP, d) <= PAINT_RADIUS_FT:
            helpers += 1
        dl = _point_to_segment_distance(d, holder, BASKETBALL_HOOP) * BASKET_FT_PER_UNIT
        if dl < lane:
            lane = dl
    return [
        d_ft, d_ft * d_ft / 100.0, on_ball, lane, float(helpers), lane * on_ball, d_ft * helpers / 10.0,
        max(0.0, 4.0 - on_ball), max(0.0, 4.0 - lane), float(min(helpers, 3)),
    ]


def basketball_features(head: str, holder: Point, defenders: Sequence[Point],
                        receiver: Point = None) -> List[float]:
    """Vector de caracteristicas de una cabeza de baloncesto.

    head: "shot" (tiro del portador), "pass" (pase al `receiver`) o
    "drive" (penetracion al aro). `defenders` son las posiciones de los
    defensores en el momento de la accion (en inferencia: las PREDICHAS tras
    la reaccion defensiva; en entrenamiento: las del dataset).
    """
    if head == "shot":
        return _basketball_shot(holder, defenders)
    if head == "pass":
        if receiver is None:
            raise ValueError("La cabeza 'pass' necesita el receptor")
        return _basketball_pass(holder, receiver, defenders)
    if head == "drive":
        return _basketball_drive(holder, defenders)
    raise ValueError(f"Cabeza de baloncesto desconocida: {head!r}")


# ----------------------------------------------------------------------------
# Tenis
# ----------------------------------------------------------------------------
def tennis_serve_features(server: Point, returner: Point, target: Point) -> List[float]:
    """Caracteristicas del saque: sacador, restador (ya con su anticipacion) y punto de bote."""
    sx, sy = server
    rx, ry = returner
    tx, ty = target

    # Donde cruzaria la pelota la altura (Y) del restador si siguiera en linea
    # recta desde el sacador, pasando por el punto de bote. La diferencia
    # lateral con el restador es cuanto debe moverse de lado para pegarle.
    dy = ty - sy
    if abs(dy) < 1e-9:
        arrival_x = tx
    else:
        arrival_x = tx + (tx - sx) * (ry - ty) / dy
    arrival_x = max(-50.0, min(150.0, arrival_x))
    miss = min(30.0, abs(arrival_x - rx) * TENNIS_FT_PER_X)

    run = math.hypot((tx - rx) * TENNIS_FT_PER_X, (ty - ry) * TENNIS_FT_PER_Y)
    length = math.hypot((tx - sx) * TENNIS_FT_PER_X, (ty - sy) * TENNIS_FT_PER_Y)
    side = min(13.5, max(0.0, min(tx, 100.0 - tx)) * TENNIS_FT_PER_X)
    center = min(13.5, abs(tx - 50.0) * TENNIS_FT_PER_X)
    depth = max(-10.0, min(17.0, (ty - TENNIS_SERVICE_LINE_Y) * TENNIS_FT_PER_Y))
    out_box = 1.0 if (ty < TENNIS_SERVICE_LINE_Y or ty > TENNIS_NET_Y) else 0.0
    back = max(0.0, ry) * TENNIS_FT_PER_Y

    short = max(0.0, min(depth, 16.7)) / 16.7          # 0 = profundo, 1 = corto
    wide = (13.5 - side) / 13.5                        # 0 = al centro, 1 = pegado a la linea lateral
    vec = [miss, miss * miss / 10.0, run, length, side, center, depth, out_box, back, miss * short, miss * wide,
           max(0.0, miss - 7.0), max(0.0, miss - 12.0)]
    if out_box:
        # Un saque que cae fuera de la caja es falta casi segura: la geometria
        # fina deja de importar. Se anulan todas las caracteristicas salvo la
        # bandera `fuera_de_caja` (indice 7) para que el modelo lineal no
        # "extrapole" pesos pensados para saques validos.
        vec = [v if i == 7 else 0.0 for i, v in enumerate(vec)]
    return vec
