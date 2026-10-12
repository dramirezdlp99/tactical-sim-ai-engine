"""
Guardar en: tactical-sim-ai-engine/tests/test_modelo.py

Pruebas del Modelo Predictivo y del "Cuerpo Tecnico Rival":
  1. Paridad de caracteristicas (entrenamiento == produccion).
  2. Sanidad de monotonia (el modelo "razona" en la direccion correcta).
  3. Invariantes de la respuesta defensiva.
  4. Fallback cuando faltan o estan danados los pesos.
  5. Pruebas de extremo a extremo de process_simulation_async.
  6. Entrenamiento, pesos entregados y lectura de CSV real.

Ejecutar desde la raiz del repo:  python -m pytest tests -v
"""

import asyncio
import json
import math
import os
import random

import pytest

from app.ml import features as F
from app.schemas.simulation import TacticalSimulationResponse
from app.services import ai_engine, defense_response as DR, trained_model


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def payload(att, dfn, ball):
    return {
        "match_id": "m1",
        "team_home_positions": [{"player_id": f"J{i + 1}", "x": x, "y": y} for i, (x, y) in enumerate(att)],
        "team_away_positions": [{"player_id": f"D{i + 1}", "x": x, "y": y} for i, (x, y) in enumerate(dfn)],
        "ball_position": {"player_id": "BALL", "x": ball[0], "y": ball[1]},
    }


def tennis_payload(rx=50.0, ry=10.0, target=(30.0, 38.0), server=(50.0, 90.0)):
    return {
        "match_id": "t1",
        "team_home_positions": [{"player_id": "S", "x": server[0], "y": server[1]}],
        "team_away_positions": [{"player_id": "R", "x": rx, "y": ry}],
        "ball_position": {"player_id": "BALL", "x": target[0], "y": target[1]},
    }


# Jugadas de referencia cuyo resultado con las FORMULAS ANTIGUAS se midio con el
# ai_engine.py original (antes de este cambio). Sirven para comprobar el fallback.
JUGADA_TIRO = payload([(50, 25), (20, 35), (80, 35), (30, 60), (70, 60)],
                      [(60, 40), (24, 30), (76, 32), (35, 55), (66, 58)], (50, 25))
JUGADA_PENETRACION = payload([(50, 70), (20, 35), (80, 35), (30, 60), (70, 60)],
                             [(50, 60), (24, 50), (76, 32), (35, 70), (66, 80)], (50, 70))


def random_basketball_state(rng):
    att = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(5)]
    dfn = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(5)]
    ball = (rng.uniform(0, 100), rng.uniform(0, 100))
    return att, dfn, ball


# ---------------------------------------------------------------------------
# 1. Paridad de caracteristicas
# ---------------------------------------------------------------------------
def test_longitud_de_vectores_coincide_con_nombres():
    h, d = (50.0, 25.0), [(52.0, 30.0), (10.0, 10.0)]
    assert len(F.basketball_features("shot", h, d)) == len(F.SHOT_FEATURES)
    assert len(F.basketball_features("pass", h, d, (20.0, 40.0))) == len(F.PASS_FEATURES)
    assert len(F.basketball_features("drive", h, d)) == len(F.DRIVE_FEATURES)
    assert len(F.tennis_serve_features((50, 90), (50, 10), (30, 38))) == len(F.SERVE_FEATURES)
    # fuera de la caja tambien debe mantener la longitud
    assert len(F.tennis_serve_features((50, 90), (50, 10), (30, 20))) == len(F.SERVE_FEATURES)


def test_pesos_entregados_usan_los_mismos_nombres_que_features():
    for sport, heads in F.FEATURE_NAMES.items():
        raw = json.load(open(trained_model.MODELS_DIR / f"{sport}_v1.json", encoding="utf-8"))
        assert raw["feature_schema_version"] == F.FEATURE_SCHEMA_VERSION
        for head, names in heads.items():
            assert raw["heads"][head]["feature_names"] == names


def test_produccion_usa_exactamente_las_caracteristicas_de_features(monkeypatch):
    """Espia lo que ai_engine le pasa al modelo y lo compara con features.py aplicado a la defensa predicha."""
    modelo = trained_model.load_model("basketball")
    visto = []
    original = modelo.predict
    monkeypatch.setattr(modelo, "predict", lambda head, feats: (visto.append((head, list(feats))), original(head, feats))[1])

    data = JUGADA_TIRO
    ai_engine._simulate_basketball(data)

    attackers = [(p["player_id"], (p["x"], p["y"])) for p in data["team_home_positions"]]
    defenders = [(p["player_id"], (p["x"], p["y"])) for p in data["team_away_positions"]]
    ball = (data["ball_position"]["x"], data["ball_position"]["y"])
    h = F.basketball_holder_index([a[1] for a in attackers], ball)
    reaccion = DR.predict_basketball_defense(attackers, defenders, ball, h)
    def_xy = [p[1] for p in reaccion.positions]
    esperado = F.basketball_features("shot", attackers[h][1], def_xy)
    tiros = [f for head, f in visto if head == "shot"]
    assert tiros and tiros[0] == esperado


def test_produccion_tenis_usa_exactamente_las_caracteristicas(monkeypatch):
    modelo = trained_model.load_model("tennis")
    visto = []
    original = modelo.predict
    monkeypatch.setattr(modelo, "predict", lambda head, feats: (visto.append(list(feats)), original(head, feats))[1])
    d = tennis_payload()
    ai_engine._simulate_tennis(d)
    nueva = DR.predict_tennis_returner((50, 90), ("R", (50, 10)), (30, 38)).positions[0][1]
    assert visto[0] == F.tennis_serve_features((50, 90), nueva, (30, 38))


def test_csv_real_produce_las_mismas_caracteristicas(tmp_path):
    pytest.importorskip("numpy")
    from app.ml import dataset as D
    att = [(50, 25), (20, 35), (80, 35), (30, 60), (70, 60)]
    dfn = [(60, 40), (24, 30), (76, 32), (35, 55), (66, 58)]
    cols = ["head", "outcome", "ball_x", "ball_y", "receiver"]
    cols += [f"a{i}_{c}" for i in range(1, 6) for c in "xy"] + [f"d{i}_{c}" for i in range(1, 6) for c in "xy"]
    row = {"head": "pass", "outcome": 1, "ball_x": 50, "ball_y": 25, "receiver": 3}
    for i in range(5):
        row[f"a{i + 1}_x"], row[f"a{i + 1}_y"] = att[i]
        row[f"d{i + 1}_x"], row[f"d{i + 1}_y"] = dfn[i]
    p = tmp_path / "real.csv"
    p.write_text(",".join(cols) + "\n" + ",".join(str(row.get(c, "")) for c in cols) + "\n", encoding="utf-8")
    datos = D.load_csv(str(p), "basketball")
    assert list(datos["pass"]["X"][0]) == F.basketball_features("pass", att[0], dfn, att[2])
    assert list(datos["pass"]["y"]) == [1.0]

    t = tmp_path / "tenis.csv"
    t.write_text("outcome,server_x,server_y,returner_x,returner_y,target_x,target_y\n0,50,90,50,10,30,38\n", encoding="utf-8")
    dt = D.load_csv(str(t), "tennis")
    assert list(dt["serve"]["X"][0]) == F.tennis_serve_features((50, 90), (50, 10), (30, 38))


# ---------------------------------------------------------------------------
# 2. Sanidad de monotonia
# ---------------------------------------------------------------------------
def test_modelo_tiro_defensor_mas_cerca_baja_la_probabilidad():
    holder = (50.0, 22.0)
    prev = None
    for d_units in [24, 18, 14, 10, 7, 5, 3, 1.5]:           # defensor cada vez mas cerca (detras del tirador)
        p = trained_model.predict("basketball", "shot", F.basketball_features("shot", holder, [(50.0, 22.0 + d_units)]))
        if prev is not None:
            assert p <= prev + 1e-9
        prev = p
    lejos = trained_model.predict("basketball", "shot", F.basketball_features("shot", holder, [(50.0, 46.0)]))
    cerca = trained_model.predict("basketball", "shot", F.basketball_features("shot", holder, [(50.0, 23.5)]))
    assert lejos - cerca > 0.05


def test_modelo_tiro_mas_cerca_del_aro_sube_la_probabilidad():
    defensores = [(10.0, 60.0)]
    p_aro = trained_model.predict("basketball", "shot", F.basketball_features("shot", (50.0, 14.0), defensores))
    p_media = trained_model.predict("basketball", "shot", F.basketball_features("shot", (50.0, 30.0), defensores))
    p_triple = trained_model.predict("basketball", "shot", F.basketball_features("shot", (50.0, 62.0), defensores))
    assert p_aro > p_media > p_triple


def test_modelo_pase_carril_bloqueado_baja_la_probabilidad():
    pasador, receptor = (50.0, 40.0), (50.0, 70.0)
    libre = trained_model.predict("basketball", "pass", F.basketball_features("pass", pasador, [(10.0, 10.0)], receptor))
    bloqueado = trained_model.predict("basketball", "pass", F.basketball_features("pass", pasador, [(50.0, 55.0)], receptor))
    assert libre - bloqueado > 0.15


def test_e2e_tiro_defensor_mas_cerca_baja_el_exito():
    # Tirador J1 a 15 unidades del aro; D1 cubre al tirador desde atras; el resto marca a sus hombres.
    def jugada(dist_def):
        att = [(50, 25), (15, 60), (85, 60), (15, 85), (85, 85)]
        dfn = [(50, 25 + dist_def), (18, 60), (82, 60), (18, 85), (82, 85)]
        return payload(att, dfn, (50, 25))
    exitos = []
    for dist in (34, 28, 22):
        r = ai_engine._simulate_basketball(jugada(dist))
        assert r["recommended_action"] == "TIRO_INMEDIATO"
        exitos.append(r["success_probability"])
    assert exitos[0] > exitos[1] > exitos[2]


def test_e2e_tiro_mas_cerca_del_aro_sube_el_exito():
    def jugada(y):
        att = [(50, y), (15, 70), (85, 70), (15, 90), (85, 90)]
        dfn = [(50, y + 30), (18, 70), (82, 70), (18, 90), (82, 90)]
        return payload(att, dfn, (50, y))
    cerca = ai_engine._simulate_basketball(jugada(18))
    lejos = ai_engine._simulate_basketball(jugada(33))
    assert cerca["recommended_action"] == lejos["recommended_action"] == "TIRO_INMEDIATO"
    assert cerca["success_probability"] > lejos["success_probability"]


def test_e2e_tenis_restador_mas_lejos_sube_la_probabilidad():
    # Saque desde (50,90) al punto (30,38): la pelota llega al fondo por x ~ 19.
    p = [ai_engine._simulate_tennis(tennis_payload(rx=rx))["success_probability"] for rx in (20, 35, 50, 65, 80)]
    assert p[0] < p[1] < p[2] < p[3] < p[4]


# ---------------------------------------------------------------------------
# 3. Invariantes de defense_response
# ---------------------------------------------------------------------------
def test_defensa_dentro_de_cancha_y_paso_maximo_respetado():
    rng = random.Random(7)
    for _ in range(400):
        att_xy, dfn_xy, ball = random_basketball_state(rng)
        att = [(f"J{i}", p) for i, p in enumerate(att_xy)]
        dfn = [(f"D{i}", p) for i, p in enumerate(dfn_xy)]
        h = F.basketball_holder_index(att_xy, ball)
        r = DR.predict_basketball_defense(att, dfn, ball, h)
        assert [p[0] for p in r.positions] == [d[0] for d in dfn]          # mismo orden e ids
        assert isinstance(r.scheme, str) and r.scheme.startswith("Hombre a hombre")
        for (_, nuevo), (_, viejo) in zip(r.positions, dfn):
            assert 0.0 <= nuevo[0] <= 100.0 and 0.0 <= nuevo[1] <= 100.0
            assert math.hypot(nuevo[0] - viejo[0], nuevo[1] - viejo[1]) <= DR.MAX_REACTION_STEP + 1e-9


def test_emparejamiento_voraz_es_uno_a_uno():
    rng = random.Random(3)
    for _ in range(200):
        att = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(5)]
        dfn = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(5)]
        marcas = DR.greedy_match(dfn, att)
        assert sorted(marcas) == [0, 1, 2, 3, 4]
    # mas defensores que atacantes: sobran defensores (-1)
    marcas = DR.greedy_match([(0, 0), (10, 10), (90, 90)], [(1, 1), (89, 89)])
    assert sorted(marcas) == [-1, 0, 1]


def test_defensor_se_acerca_al_lado_del_aro_de_su_marca():
    att = [("J1", (50.0, 60.0))] + [(f"J{i}", (10.0 + 20 * i, 90.0)) for i in range(2, 6)]
    dfn = [("D1", (50.0, 75.0))] + [(f"D{i}", (10.0 + 20 * i, 95.0)) for i in range(2, 6)]
    r = DR.predict_basketball_defense(att, dfn, (50.0, 60.0), 0)
    nuevo = r.positions[0][1]
    assert nuevo[1] < 75.0 and abs(nuevo[1] - 60.0) < 15.0 and abs(nuevo[0] - 50.0) < 1e-6


def test_esquema_con_ayuda_solo_si_el_balon_esta_cerca_del_aro():
    att = [("J1", (50.0, 20.0)), ("J2", (15.0, 40.0)), ("J3", (85.0, 40.0)), ("J4", (30.0, 70.0)), ("J5", (70.0, 70.0))]
    dfn = [("D1", (50.0, 25.0)), ("D2", (18.0, 42.0)), ("D3", (82.0, 42.0)), ("D4", (32.0, 68.0)), ("D5", (68.0, 68.0))]
    cerca = DR.predict_basketball_defense(att, dfn, (50.0, 20.0), 0)
    assert cerca.scheme == "Hombre a hombre con ayuda en la pintura"
    att_lejos = [("J1", (50.0, 70.0))] + att[1:]
    lejos = DR.predict_basketball_defense(att_lejos, dfn, (50.0, 70.0), 0)
    assert lejos.scheme == "Hombre a hombre"


def test_restador_tenis_anticipa_dentro_de_limites():
    rng = random.Random(11)
    for _ in range(300):
        rx, ry = rng.uniform(0, 100), rng.uniform(0, 30)
        tx, ty = rng.uniform(0, 100), rng.uniform(28.6, 50)
        r = DR.predict_tennis_returner((50, 90), ("R", (rx, ry)), (tx, ty))
        nx, ny = r.positions[0][1]
        assert 0 <= nx <= 100 and 0 <= ny <= 100
        paso_ft = math.hypot((nx - rx) * 0.27, (ny - ry) * 0.78)
        assert paso_ft <= DR.TENNIS_MAX_ANTICIPATION_FT + 1e-9
        assert abs(nx - tx) <= abs(rx - tx) + 1e-9            # se acerca (lateralmente) al bote
        assert r.scheme in (DR.SCHEME_RETURN_ANTICIPATE, DR.SCHEME_RETURN_WAIT)


# ---------------------------------------------------------------------------
# 4. Fallback
# ---------------------------------------------------------------------------
def _sin_modelos(monkeypatch, tmp_path):
    monkeypatch.setattr(trained_model, "MODELS_DIR", tmp_path)
    trained_model.reset_cache()


def test_fallback_sin_pesos_baloncesto_igual_a_formulas_antiguas(monkeypatch, tmp_path):
    _sin_modelos(monkeypatch, tmp_path)
    assert trained_model.load_model("basketball") is None
    r = ai_engine._simulate_basketball(JUGADA_TIRO)
    assert r["success_probability"] == 0.7523 and r["recommended_action"] == "TIRO_INMEDIATO"
    assert r["risk_index"] == 0.0 and r["secondary_efficiency"] == 0.8939
    assert r["clean_shot_probability"] is None and r["pass_effectiveness"] is None
    assert r["predicted_defense"] == [] and "no disponible" in r["model_info"]
    r2 = ai_engine._simulate_basketball(JUGADA_PENETRACION)
    assert r2["success_probability"] == 0.4409 and r2["recommended_action"] == "PENETRAR_AL_ARO"
    assert ai_engine._simulate_basketball_prob_only(JUGADA_TIRO)["success_probability"] == pytest.approx(0.7523, abs=1e-4)


def test_fallback_sin_pesos_tenis_igual_a_formulas_antiguas(monkeypatch, tmp_path):
    _sin_modelos(monkeypatch, tmp_path)
    r = ai_engine._simulate_tennis(tennis_payload())
    assert r["success_probability"] == 0.88 and r["recommended_action"] == "ACE_PROBABLE"
    assert r["risk_index"] == 0.2 and r["secondary_efficiency"] == 0.987
    assert "no disponible" in r["model_info"]


def test_fallback_con_json_corrupto_o_esquema_distinto(monkeypatch, tmp_path):
    (tmp_path / "basketball_v1.json").write_text("{esto no es json", encoding="utf-8")
    raw = json.load(open(trained_model.MODELS_DIR / "tennis_v1.json", encoding="utf-8"))
    raw["heads"]["serve"]["feature_names"][0] = "nombre_cambiado"      # simula training-serving skew
    (tmp_path / "tennis_v1.json").write_text(json.dumps(raw), encoding="utf-8")
    _sin_modelos(monkeypatch, tmp_path)
    assert trained_model.load_model("basketball") is None and trained_model.load_model("tennis") is None
    assert ai_engine._simulate_basketball(JUGADA_TIRO)["success_probability"] == 0.7523
    assert ai_engine._simulate_tennis(tennis_payload())["success_probability"] == 0.88


def test_fallback_si_el_modelo_falla_a_mitad_de_prediccion(monkeypatch):
    modelo = trained_model.load_model("basketball")

    def explota(head, feats):
        raise RuntimeError("fallo simulado")
    monkeypatch.setattr(modelo, "predict", explota)
    r = ai_engine._simulate_basketball(JUGADA_TIRO)
    assert r["success_probability"] == 0.7523 and r["clean_shot_probability"] is None
    assert r["predicted_defense"] == [] and "no disponible" in r["model_info"]


def test_nombres_publicos_de_ai_engine_se_conservan():
    for nombre in ("_simulate_basketball", "_simulate_tennis", "_distance", "_point_to_segment_distance",
                   "_clamp", "_compute_open_space_grid", "BASKETBALL_HOOP", "DEFENSE_PRESSURE_RADIUS",
                   "OPEN_LANE_THRESHOLD", "SHOOT_RANGE", "COURT_DIAGONAL", "MAX_RETURNER_REACH",
                   "IDEAL_SERVE_TRAVEL", "_run_simulation_sync", "AIEngineService", "ai_engine_service"):
        assert hasattr(ai_engine, nombre), nombre


# ---------------------------------------------------------------------------
# 5. Extremo a extremo
# ---------------------------------------------------------------------------
@pytest.fixture
def pocas_iteraciones(monkeypatch):
    from app.services import monte_carlo
    monkeypatch.setattr(monte_carlo, "MONTE_CARLO_ITERATIONS", 150)


def test_e2e_baloncesto_process_simulation_async(pocas_iteraciones):
    r = asyncio.run(ai_engine.ai_engine_service.process_simulation_async(JUGADA_PENETRACION))
    resp = TacticalSimulationResponse(**r)                 # valida contra el esquema real
    assert 0.0 <= resp.success_probability <= 1.0
    assert resp.recommended_action in ("TIRO_INMEDIATO", "REPOSICIONAR_Y_ESPERAR", "PENETRAR_AL_ARO") \
        or resp.recommended_action.startswith("PASE_A_")
    assert len(resp.predicted_defense) == 5
    assert all(0 <= d.x <= 100 and 0 <= d.y <= 100 for d in resp.predicted_defense)
    assert resp.defense_scheme.startswith("Hombre a hombre")
    assert 0.0 <= resp.clean_shot_probability <= 1.0 and 0.0 <= resp.pass_effectiveness <= 1.0
    assert 0.0 <= resp.risk_index <= 1.0 and resp.heat_map_coords
    assert "Regresion logistica v1" in resp.model_info and "AUC" in resp.model_info
    assert "Al promediar 150 variaciones realistas" in resp.tactical_note
    assert resp.execution_thread.startswith("MultiProcess-SharedMemory(workers=")


def test_e2e_tenis_process_simulation_async(pocas_iteraciones):
    r = asyncio.run(ai_engine.ai_engine_service.process_simulation_async(tennis_payload()))
    resp = TacticalSimulationResponse(**r)
    assert 0.0 < resp.success_probability < 1.0
    assert len(resp.predicted_defense) == 1 and resp.predicted_defense[0].player_id == "R"
    assert 0 <= resp.predicted_defense[0].x <= 100 and 0 <= resp.predicted_defense[0].y <= 100
    assert resp.defense_scheme and resp.clean_shot_probability is None and resp.pass_effectiveness is None
    assert resp.ball_speed_kmh is not None and resp.spin_rate_rpm is not None
    assert "AUC" in resp.model_info and "Al promediar 150 variaciones realistas" in resp.tactical_note


def test_e2e_a_traves_de_un_endpoint_fastapi(pocas_iteraciones):
    """Reproduce el contrato del endpoint real: valida entrada y salida con los esquemas."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.schemas.simulation import TacticalSimulationRequest

    app = FastAPI()

    @app.post("/sim", response_model=TacticalSimulationResponse)
    async def sim(req: TacticalSimulationRequest):
        return await ai_engine.ai_engine_service.process_simulation_async(req.model_dump())

    cliente = TestClient(app)
    for cuerpo in (JUGADA_TIRO, tennis_payload()):
        res = cliente.post("/sim", json=cuerpo)
        assert res.status_code == 200, res.text
        js = res.json()
        for campo in ("predicted_defense", "defense_scheme", "clean_shot_probability", "pass_effectiveness", "model_info"):
            assert campo in js


def test_monte_carlo_usa_la_version_ligera():
    """Las iteraciones solo deben calcular la probabilidad (sin mapa de calor ni notas)."""
    ligera = ai_engine._simulate_basketball_prob_only(JUGADA_TIRO)
    assert set(ligera) == {"success_probability"}
    completa = ai_engine._simulate_basketball(JUGADA_TIRO)
    assert ligera["success_probability"] == pytest.approx(completa["success_probability"], abs=1e-4)
    t_lig = ai_engine._simulate_tennis_prob_only(tennis_payload())
    t_com = ai_engine._simulate_tennis(tennis_payload())
    assert set(t_lig) == {"success_probability"}
    assert t_lig["success_probability"] == pytest.approx(t_com["success_probability"], abs=1e-4)


def test_worker_de_monte_carlo_prefiere_la_version_ligera_y_es_compatible_hacia_atras(monkeypatch):
    import struct
    from multiprocessing import shared_memory
    from app.services import monte_carlo

    def correr(n=5):
        shm = shared_memory.SharedMemory(create=True, size=n * 8)
        try:
            monte_carlo._worker_run_chunk(shm.name, 0, n, JUGADA_TIRO, "_simulate_basketball", 1)
            return struct.unpack(f"{n}d", bytes(shm.buf[: n * 8]))
        finally:
            shm.close()
            shm.unlink()

    llamadas = []
    monkeypatch.setattr(ai_engine, "_simulate_basketball_prob_only", lambda d: (llamadas.append(1), {"success_probability": 0.3})[1])
    monkeypatch.setattr(ai_engine, "_simulate_basketball", lambda d: pytest.fail("no debe usar la version completa"))
    assert correr() == (0.3,) * 5 and len(llamadas) == 5

    # Sin version ligera, usa la funcion completa de siempre (interfaz original).
    monkeypatch.delattr(ai_engine, "_simulate_basketball_prob_only")
    monkeypatch.setattr(ai_engine, "_simulate_basketball", lambda d: {"success_probability": 0.7})
    assert correr() == (0.7,) * 5


def test_datos_insuficientes_no_rompe():
    r = ai_engine._simulate_basketball(payload([(50, 50)], [], (50, 50)))
    assert r["recommended_action"] == "DATOS_INSUFICIENTES"
    TacticalSimulationResponse(**r)


def test_un_solo_atacante_y_defensores_de_mas_no_rompe():
    r = ai_engine._simulate_basketball(payload([(50, 30)], [(50, 40), (60, 40), (40, 40)], (50, 30)))
    TacticalSimulationResponse(**r)
    assert 0.0 <= r["success_probability"] <= 1.0


# ---------------------------------------------------------------------------
# 6. Entrenamiento y pesos entregados
# ---------------------------------------------------------------------------
def test_pesos_entregados_son_validos_pequenos_y_con_metadatos():
    for sport in ("basketball", "tennis"):
        ruta = trained_model.MODELS_DIR / f"{sport}_v1.json"
        assert ruta.stat().st_size < 100 * 1024
        raw = json.load(open(ruta, encoding="utf-8"))
        for clave in ("version", "n_samples", "seed", "trained_at", "synthetic_data_warning", "data_source", "heads"):
            assert clave in raw
        assert raw["n_samples"] >= 60_000 and raw["data_source"] == "synthetic"
        assert "SINTETICOS" in raw["synthetic_data_warning"]
        for head in raw["heads"].values():
            m = head["metrics"]
            assert m["modelo"]["auc"] > 0.55
            assert m["modelo"]["log_loss"] < m["baseline_constante"]["log_loss"]      # mejor que el baseline
            assert m["modelo"]["brier"] < m["baseline_constante"]["brier"]
            assert m["modelo"]["ece"] < 0.03
        assert trained_model.load_model(sport) is not None


def test_dataset_es_determinista_con_la_misma_semilla():
    pytest.importorskip("numpy")
    import numpy as np
    from app.ml import dataset as D
    a = D.generate_basketball(300, seed=5)
    b = D.generate_basketball(300, seed=5)
    c = D.generate_basketball(300, seed=6)
    assert np.array_equal(a["shot"]["X"], b["shot"]["X"]) and np.array_equal(a["pass"]["y"], b["pass"]["y"])
    assert not np.array_equal(a["shot"]["X"], c["shot"]["X"])
    t1, t2 = D.generate_tennis(300, seed=5), D.generate_tennis(300, seed=5)
    assert np.array_equal(t1["serve"]["X"], t2["serve"]["X"])


def test_los_resultados_son_binarios_y_la_verdad_es_plausible():
    pytest.importorskip("numpy")
    from app.ml import dataset as D
    b = D.generate_basketball(4000, seed=1)
    for d in b.values():
        assert set(d["y"].tolist()) <= {0.0, 1.0}
    # Porcentajes cerca del aro / media distancia / triple con defensor lejos (verdad de terreno sin ruido)
    rng = random.Random(0)
    def media(holder, n=300):
        return sum(D.shot_truth(holder, [(holder[0], holder[1] + 30)], rng)[0] for _ in range(n)) / n
    assert 0.58 <= media((50, 14)) <= 0.68           # aro: 60-65% aprox.
    assert 0.38 <= media((50, 40)) <= 0.48           # media distancia (15 pies) ~40-45%
    assert 0.33 <= media((50, 60)) <= 0.40           # triple ~36%


def test_entrenamiento_corto_genera_pesos_validos(tmp_path):
    pytest.importorskip("numpy")
    from app.ml import train
    assert train.main(["--samples", "3000", "--seed", "3", "--out-dir", str(tmp_path)]) == 0
    for sport in ("basketball", "tennis"):
        raw = json.load(open(tmp_path / f"{sport}_v1.json", encoding="utf-8"))
        modelo = trained_model._parse(sport, raw)            # pasa la validacion del cargador
        assert set(modelo.heads) == set(F.FEATURE_NAMES[sport])
        for h in raw["heads"].values():
            assert h["metrics"]["modelo"]["log_loss"] < h["metrics"]["baseline_constante"]["log_loss"] + 0.01


def test_metricas_auc_y_calibracion_basicas():
    pytest.importorskip("numpy")
    import numpy as np
    from app.ml import train
    y = np.array([0, 0, 1, 1], dtype=float)
    assert train.auc_score(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert train.auc_score(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert train.auc_score(y, np.full(4, 0.5)) == 0.5
    assert train.ece(np.array([1.0, 0.0, 1.0, 0.0]), np.full(4, 0.5)) == 0.0
