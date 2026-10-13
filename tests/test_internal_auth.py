"""
Guardar en: tactical-sim-ai-engine/tests/test_internal_auth.py

Pruebas de la clave interna entre el backend y el motor de IA.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

BODY = {
    "match_id": "t-1",
    "team_home_positions": [{"player_id": f"j{i}", "x": 20 + i * 12, "y": 60} for i in range(1, 6)],
    "team_away_positions": [{"player_id": f"d{i}", "x": 24 + i * 12, "y": 52} for i in range(1, 6)],
    "ball_position": {"player_id": "ball", "x": 44, "y": 60},
}
URL = "/api/v1/simulation/predict"


@pytest.fixture
def client():
    return TestClient(app)


def test_sin_clave_configurada_no_se_exige_nada(client, monkeypatch):
    monkeypatch.delenv("AI_INTERNAL_KEY", raising=False)
    assert client.post(URL, json=BODY).status_code == 200


def test_con_clave_configurada_rechaza_sin_cabecera(client, monkeypatch):
    monkeypatch.setenv("AI_INTERNAL_KEY", "secreto-de-prueba")
    assert client.post(URL, json=BODY).status_code == 401


def test_con_clave_configurada_rechaza_clave_incorrecta(client, monkeypatch):
    monkeypatch.setenv("AI_INTERNAL_KEY", "secreto-de-prueba")
    assert client.post(URL, json=BODY, headers={"X-Internal-Key": "otra"}).status_code == 401


def test_con_clave_correcta_responde(client, monkeypatch):
    monkeypatch.setenv("AI_INTERNAL_KEY", "secreto-de-prueba")
    r = client.post(URL, json=BODY, headers={"X-Internal-Key": "secreto-de-prueba"})
    assert r.status_code == 200
    assert 0 <= r.json()["success_probability"] <= 1


def test_health_y_resumen_del_modelo_siguen_publicos(client, monkeypatch):
    monkeypatch.setenv("AI_INTERNAL_KEY", "secreto-de-prueba")
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/model/info").status_code == 200
