"""
Guardar en: tactical-sim-ai-engine/tests/test_model_info.py

Pruebas del resumen publico del modelo (GET /api/v1/model/info).
"""
import json

import pytest

from app.services import model_summary


def test_resumen_incluye_ambos_deportes_y_las_metricas():
    s = model_summary.summary()
    assert set(s["models"]) == {"basketball", "tennis"}
    bb = s["models"]["basketball"]
    assert bb["available"] is True
    assert set(bb["heads"]) == {"shot", "pass", "drive"}
    assert bb["n_samples"] == 80000
    for head in bb["heads"].values():
        assert 0.5 < head["auc"] < 1.0
        assert head["auc_baseline"] == 0.5
        assert head["n_features"] > 0
    assert set(s["models"]["tennis"]["heads"]) == {"serve"}


def test_el_resumen_avisa_que_los_datos_son_sinteticos():
    s = model_summary.summary()
    for sport in ("basketball", "tennis"):
        m = s["models"][sport]
        assert m["data_source"] == "synthetic"
        assert "SINTETICOS" in m["synthetic_data_warning"]


def test_el_resumen_no_expone_pesos_ni_coeficientes():
    texto = json.dumps(model_summary.summary())
    for prohibido in ("weights", "coef", "intercept", "bias", "mean", "std"):
        assert f'"{prohibido}"' not in texto


def test_endpoint_http_responde_el_resumen():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.v1.endpoints import model_info

    app = FastAPI()
    app.include_router(model_info.router, prefix="/api/v1/model")
    r = TestClient(app).get("/api/v1/model/info")
    assert r.status_code == 200
    body = r.json()
    assert "generated_at" in body
    assert body["models"]["basketball"]["available"] is True


def test_si_no_hay_modelo_se_informa_no_disponible(monkeypatch):
    from app.services import trained_model
    monkeypatch.setattr(trained_model, "load_model", lambda sport: None)
    s = model_summary.summary()
    assert s["models"]["basketball"] == {"available": False}
