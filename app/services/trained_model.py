"""
Guardar en: tactical-sim-ai-engine/app/services/trained_model.py

CARGADOR Y PREDICTOR DEL MODELO ENTRENADO (sin numpy).

Lee los pesos que dejo el entrenamiento (app/ml/models/<deporte>_v1.json) y
calcula la prediccion de una regresion logistica con matematica de Python
puro:

    probabilidad = 1 / (1 + e^-(sesgo + suma(peso_i * (valor_i - media_i) / desviacion_i)))

Para que sea rapido (se llama miles de veces por peticion en Monte Carlo),
al cargar se precalcula la forma equivalente

    z = intercepto + suma(coef_i * valor_i)

Reglas de seguridad ("el motor nunca debe caerse"):
  - Si el archivo no existe, esta corrupto, tiene numeros invalidos o sus
    nombres de caracteristicas ya no coinciden con app/ml/features.py
    (training-serving skew), el modelo se considera NO disponible: load_model
    devuelve None, se registra un aviso UNA sola vez y ai_engine.py usa sus
    formulas geometricas de siempre.
  - El resultado de la carga (bueno o malo) se guarda en memoria: el JSON se
    lee del disco una sola vez por proceso.
"""

import json
import logging
import math
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app.ml import features as F

logger = logging.getLogger(__name__)

# app/services/trained_model.py -> app/ml/models/
MODELS_DIR = Path(__file__).resolve().parent.parent / "ml" / "models"
MODEL_VERSION = "v1"

_HEAD_LABELS = {"shot": "tiro", "pass": "pase", "drive": "penetracion", "serve": "saque"}


class HeadModel:
    """Una cabeza (por ejemplo "tiro"): regresion logistica ya aplanada para evaluar rapido."""

    __slots__ = ("name", "n_features", "coef", "intercept", "metrics")

    def __init__(self, name: str, coef: List[float], intercept: float, metrics: dict):
        self.name = name
        self.coef = coef
        self.intercept = intercept
        self.n_features = len(coef)
        self.metrics = metrics

    def prob(self, feats: Sequence[float]) -> float:
        z = self.intercept
        for c, v in zip(self.coef, feats):
            z += c * v
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-min(z, 60.0)))
        e = math.exp(max(z, -60.0))
        return e / (1.0 + e)


class SportModel:
    def __init__(self, sport: str, heads: Dict[str, HeadModel], meta: dict):
        self.sport = sport
        self.heads = heads
        self.meta = meta
        self.info = _build_info(sport, heads, meta)

    def predict(self, head: str, feats: Sequence[float]) -> float:
        return self.heads[head].prob(feats)


def _build_info(sport: str, heads: Dict[str, HeadModel], meta: dict) -> str:
    n = f"{int(meta.get('n_samples', 0)):,}".replace(",", ".")
    synthetic = meta.get("data_source") != "real"
    if sport == "basketball":
        unit, kind = "posesiones", ("sinteticas" if synthetic else "reales")
    else:
        unit, kind = "saques", ("sinteticos" if synthetic else "reales")
    aucs = []
    for name, h in heads.items():
        auc = h.metrics.get("modelo", {}).get("auc")
        if auc is not None:
            aucs.append((name, auc))
    if len(aucs) == 1:
        auc_txt = f"AUC {aucs[0][1]:.2f}"
    else:
        auc_txt = "AUC " + ", ".join(f"{_HEAD_LABELS.get(k, k)} {v:.2f}" for k, v in aucs)
    return f"Regresion logistica {meta.get('version', MODEL_VERSION)} entrenada con {n} {unit} {kind} · {auc_txt}"


def _parse(sport: str, raw: dict) -> SportModel:
    """Valida el JSON y construye el modelo. Lanza ValueError si algo no cuadra."""
    if raw.get("sport") != sport:
        raise ValueError(f"el archivo es de otro deporte: {raw.get('sport')!r}")
    if raw.get("feature_schema_version") != F.FEATURE_SCHEMA_VERSION:
        raise ValueError("version del esquema de caracteristicas distinta; hay que reentrenar")
    expected = F.FEATURE_NAMES[sport]
    heads: Dict[str, HeadModel] = {}
    for head, names in expected.items():
        h = raw["heads"][head]
        if list(h["feature_names"]) != list(names):
            raise ValueError(f"las caracteristicas de la cabeza {head!r} no coinciden con features.py")
        mean, std, w = h["mean"], h["std"], h["weights"]
        if not (len(mean) == len(std) == len(w) == len(names)):
            raise ValueError(f"longitudes inconsistentes en la cabeza {head!r}")
        values = list(mean) + list(std) + list(w) + [h["bias"]]
        if any((not isinstance(v, (int, float))) or not math.isfinite(v) for v in values):
            raise ValueError(f"numeros invalidos en la cabeza {head!r}")
        if any(s <= 0 for s in std):
            raise ValueError(f"desviacion no positiva en la cabeza {head!r}")
        coef = [wi / si for wi, si in zip(w, std)]
        intercept = h["bias"] - sum(c * m for c, m in zip(coef, mean))
        heads[head] = HeadModel(head, coef, intercept, h.get("metrics", {}))
    return SportModel(sport, heads, raw)


_cache: Dict[str, Optional[SportModel]] = {}
_lock = threading.Lock()


def load_model(sport: str) -> Optional[SportModel]:
    """Devuelve el modelo del deporte o None si no esta disponible (usar fallback)."""
    if sport in _cache:                      # camino rapido: sin candado
        return _cache[sport]
    with _lock:
        if sport in _cache:
            return _cache[sport]
        model: Optional[SportModel] = None
        path = MODELS_DIR / f"{sport}_{MODEL_VERSION}.json"
        try:
            with open(path, "r", encoding="utf-8") as fh:
                model = _parse(sport, json.load(fh))
        except Exception as exc:  # archivo ausente, JSON roto, esquema distinto...
            logger.warning("Modelo %s no disponible (%s: %s); se usan las formulas geometricas.",
                           sport, type(exc).__name__, exc)
        _cache[sport] = model
        return model


def predict(sport: str, head: str, feats: Sequence[float]) -> Optional[float]:
    """Probabilidad de la cabeza indicada, o None si no hay modelo."""
    model = load_model(sport)
    if model is None:
        return None
    return model.predict(head, feats)


def model_info(sport: str) -> str:
    model = load_model(sport)
    if model is None:
        return "Formulas geometricas deterministas (modelo entrenado no disponible)"
    return model.info


def reset_cache() -> None:
    """Olvida los modelos cargados (para pruebas o tras reentrenar sin reiniciar el servidor)."""
    with _lock:
        _cache.clear()
