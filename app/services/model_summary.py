"""
Guardar en: tactical-sim-ai-engine/app/services/model_summary.py

Resumen PUBLICO y de solo lectura de los modelos entrenados (para la pagina
/modelo del frontend). Nunca expone pesos ni coeficientes: solo metadatos y
metricas de validacion, y siempre incluye la advertencia de datos sinteticos.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.services import trained_model

SPORTS = ("basketball", "tennis")


def _head_summary(name: str, head: Any, raw: dict) -> Dict[str, Any]:
    raw_head = raw.get("heads", {}).get(name, {})
    metrics = head.metrics or {}
    model = metrics.get("modelo", {})
    base = metrics.get("baseline_constante", {})
    return {
        "n_features": head.n_features,
        "n_train": raw_head.get("n_train"),
        "n_valid": raw_head.get("n_valid"),
        "base_rate": raw_head.get("base_rate_train"),
        "auc": model.get("auc"),
        "auc_baseline": base.get("auc"),
        "auc_ceiling_synthetic": metrics.get("techo_auc_verdad_sintetica"),
        "log_loss": model.get("log_loss"),
        "brier": model.get("brier"),
        "ece": model.get("ece"),
    }


def _sport_summary(sport: str) -> Optional[Dict[str, Any]]:
    model = trained_model.load_model(sport)
    if model is None:
        return None
    meta = model.meta
    return {
        "available": True,
        "version": meta.get("version"),
        "model_type": meta.get("model_type"),
        "data_source": meta.get("data_source"),
        "synthetic_data_warning": meta.get("synthetic_data_warning"),
        "n_samples": meta.get("n_samples"),
        "trained_at": meta.get("trained_at"),
        "description": model.info,
        "heads": {name: _head_summary(name, h, meta) for name, h in model.heads.items()},
    }


def summary() -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for sport in SPORTS:
        s = _sport_summary(sport)
        models[sport] = s if s is not None else {"available": False}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": models,
    }
