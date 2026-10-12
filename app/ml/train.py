"""
Guardar en: tactical-sim-ai-engine/app/ml/train.py

ENTRENAMIENTO DEL MODELO (se ejecuta UNA vez, en la maquina del dueño).

Uso (desde la raiz del repo tactical-sim-ai-engine):

    pip install numpy                      # solo para entrenar, NO va al servidor
    python -m app.ml.train                 # ambos deportes, 80.000 muestras, semilla 42
    python -m app.ml.train --sport tennis --samples 120000 --seed 7
    python -m app.ml.train --sport tennis --csv datos_reales_tenis.csv   # con datos reales

Que hace:
  1. Genera datos (sinteticos, ver dataset.py) o lee un CSV real.
  2. Los divide en 80% entrenamiento / 20% validacion (al azar, con semilla).
  3. Estandariza cada caracteristica (resta la media, divide por la
     desviacion) para que todas pesen parecido durante el entrenamiento.
  4. Entrena una REGRESION LOGISTICA: un modelo que calcula
        probabilidad = 1 / (1 + e^-(b + w1*f1 + w2*f2 + ...))
     donde f1, f2... son las caracteristicas y w1, w2... los pesos que
     aprende. Se usa el metodo de Newton (IRLS), que converge en ~10
     pasos, con regularizacion L2 (un "freno" que evita pesos exagerados).
     Los terminos cuadraticos y de interaccion ya vienen en features.py.
  5. Mide el modelo SOLO con el 20% que no vio (validacion) y lo compara
     contra un baseline tonto que siempre predice la tasa promedio.
  6. Guarda pesos + metadatos en app/ml/models/<deporte>_v1.json.

Metricas que reporta:
  - AUC: probabilidad de que el modelo ponga mas alto un exito real que un
    fallo real. 0.5 = azar, 1.0 = perfecto.
  - Log-loss: castiga las probabilidades seguras pero equivocadas (menor = mejor).
  - Brier: error cuadratico medio de la probabilidad (menor = mejor).
  - ECE: error de calibracion; si el modelo dice 70%, ¿ocurre ~70% de las veces?
    (menor = mejor).
  - "techo": AUC que lograria alguien que conociera la probabilidad verdadera
    del generador. Ningun modelo puede superarlo porque el resultado final es
    un sorteo. Solo existe con datos sinteticos.
"""

import argparse
import datetime
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

from app.ml import dataset as D
from app.ml import features as F

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_VERSION = "v1"
L2_STRENGTH = 1.0          # fuerza del freno L2 (sobre la suma de perdidas; el sesgo no se frena)
TRAIN_FRACTION = 0.8
SYNTHETIC_WARNING = (
    "DATOS SINTETICOS: este modelo se entreno con partidos simulados generados por reglas "
    "de dominio (app/ml/dataset.py), no con partidos reales. Sus probabilidades son una "
    "estimacion didactica, no una medicion de rendimiento real."
)


# ---------------------------------------------------------------------------
# Regresion logistica (Newton / IRLS con L2)
# ---------------------------------------------------------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = L2_STRENGTH, iters: int = 25) -> tuple:
    """Devuelve (pesos, sesgo) para X YA estandarizada."""
    n, k = X.shape
    A = np.hstack([X, np.ones((n, 1))])          # ultima columna = sesgo
    w = np.zeros(k + 1)
    reg = np.full(k + 1, l2)
    reg[-1] = 0.0                                # el sesgo no se frena
    for _ in range(iters):
        p = _sigmoid(A @ w)
        grad = A.T @ (p - y) + reg * w
        hess = (A * (p * (1.0 - p))[:, None]).T @ A + np.diag(reg)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w[:-1], float(w[-1])


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    """AUC por rangos (con promedio en empates)."""
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = avg_rank[inverse]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Error de calibracion esperado con 10 cajas de igual ancho."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def evaluate(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {"auc": auc_score(y, p), "log_loss": log_loss(y, p), "brier": brier(y, p), "ece": ece(y, p)}


# ---------------------------------------------------------------------------
# Entrenamiento de una cabeza
# ---------------------------------------------------------------------------
def train_head(name: str, names: List[str], X: np.ndarray, y: np.ndarray, p_obs: Optional[np.ndarray],
               seed: int, l2: float) -> Dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_train = int(len(y) * TRAIN_FRACTION)
    tr, va = perm[:n_train], perm[n_train:]

    mean = X[tr].mean(axis=0)
    std = X[tr].std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)          # una caracteristica constante no divide por cero
    Z_tr, Z_va = (X[tr] - mean) / std, (X[va] - mean) / std

    w, b = fit_logistic(Z_tr, y[tr], l2)
    p_va = _sigmoid(Z_va @ w + b)

    base_rate = float(y[tr].mean())
    metrics = {"modelo": evaluate(y[va], p_va), "baseline_constante": evaluate(y[va], np.full(len(va), base_rate))}
    if p_obs is not None:
        metrics["techo_auc_verdad_sintetica"] = auc_score(y[va], p_obs[va])
    metrics["tasa_exito_validacion"] = float(y[va].mean())

    return {
        "feature_names": names,
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "weights": [float(v) for v in w],
        "bias": b,
        "n_train": int(len(tr)),
        "n_valid": int(len(va)),
        "base_rate_train": base_rate,
        "metrics": metrics,
    }


def build_model(sport: str, data: Dict[str, Dict[str, np.ndarray]], seed: int, l2: float,
                source: str, n_samples: int) -> Dict:
    heads = {}
    for head, names in F.FEATURE_NAMES[sport].items():
        d = data[head]
        heads[head] = train_head(head, names, d["X"], d["y"], d.get("p_obs"), seed, l2)
    return {
        "sport": sport,
        "version": MODEL_VERSION,
        "model_type": "logistic_regression",
        "feature_schema_version": F.FEATURE_SCHEMA_VERSION,
        "data_source": source,
        "synthetic_data_warning": SYNTHETIC_WARNING if source == "synthetic" else "",
        "n_samples": n_samples,
        "seed": seed,
        "l2": l2,
        "trained_at": datetime.date.today().isoformat(),
        "heads": heads,
    }


def print_report(model: Dict) -> None:
    print(f"\n=== {model['sport']} {model['version']}  ({model['data_source']}, {model['n_samples']} muestras, semilla {model['seed']}) ===")
    print(f"{'cabeza':<8} {'AUC':>7} {'logloss':>9} {'Brier':>8} {'ECE':>8} | {'base AUC':>8} {'logloss':>9} {'Brier':>8} {'ECE':>8} | {'techo AUC':>9}")
    for head, h in model["heads"].items():
        m, bl = h["metrics"]["modelo"], h["metrics"]["baseline_constante"]
        ceiling = h["metrics"].get("techo_auc_verdad_sintetica", float("nan"))
        print(f"{head:<8} {m['auc']:7.4f} {m['log_loss']:9.4f} {m['brier']:8.4f} {m['ece']:8.4f} | "
              f"{bl['auc']:8.4f} {bl['log_loss']:9.4f} {bl['brier']:8.4f} {bl['ece']:8.4f} | {ceiling:9.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Entrena los modelos del AI Engine de TacticAI.")
    ap.add_argument("--sport", choices=["basketball", "tennis", "all"], default="all")
    ap.add_argument("--samples", type=int, default=D.DEFAULT_SAMPLES, help="situaciones sinteticas por deporte")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l2", type=float, default=L2_STRENGTH)
    ap.add_argument("--csv", default=None, help="CSV con datos REALES (requiere --sport basketball o tennis)")
    ap.add_argument("--out-dir", default=MODELS_DIR)
    args = ap.parse_args(argv)

    sports = ["basketball", "tennis"] if args.sport == "all" else [args.sport]
    if args.csv and args.sport == "all":
        ap.error("--csv requiere indicar --sport basketball o --sport tennis")
    os.makedirs(args.out_dir, exist_ok=True)

    for sport in sports:
        if args.csv:
            data, source = D.load_csv(args.csv, sport), "real"
            n = int(min(len(v["y"]) for v in data.values()))
        else:
            gen = D.generate_basketball if sport == "basketball" else D.generate_tennis
            data, source, n = gen(args.samples, args.seed), "synthetic", args.samples
        model = build_model(sport, data, args.seed, args.l2, source, n)
        print_report(model)
        path = os.path.join(args.out_dir, f"{sport}_{MODEL_VERSION}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(model, fh, ensure_ascii=False, indent=1)
        print(f"guardado: {path}  ({os.path.getsize(path) / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
