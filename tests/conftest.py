"""
Guardar en: tactical-sim-ai-engine/tests/conftest.py

Configuracion comun de las pruebas: agrega la raiz del repo al path (para poder
hacer `import app...`) y baja las iteraciones Monte Carlo para que las pruebas
corran rapido. Si el repo ya tiene un conftest.py, copia solo estas lineas.
"""

import os
import sys

# Antes de importar monte_carlo (lee estas variables al importarse).
os.environ.setdefault("MONTE_CARLO_ITERATIONS", "200")
os.environ.setdefault("MONTE_CARLO_WORKERS", "2")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _limpiar_cache_modelo():
    """Cada prueba empieza y termina con la cache de modelos vacia (las que cambian la
    carpeta de pesos no deben contaminar a las demas)."""
    from app.services import trained_model
    trained_model.reset_cache()
    yield
    trained_model.reset_cache()
