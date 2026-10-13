"""
Guardar en: tactical-sim-ai-engine/app/core/internal_auth.py

Autenticacion ENTRE SERVICIOS (backend -> motor de IA).

Idea de capas: la IA esta "detras" del backend. El navegador nunca debe poder
llamarla directamente, aunque alguien descubra su URL. Por eso el backend
manda un secreto compartido en la cabecera X-Internal-Key y la IA lo exige en
las rutas de simulacion.

- Si la variable de entorno AI_INTERNAL_KEY NO esta definida, no se exige nada
  (comportamiento anterior: sirve para desarrollo local y para no romper un
  despliegue en curso).
- Si esta definida, toda llamada sin la clave correcta recibe 401.
- /health y /api/v1/model/info siguen publicos a proposito (no exponen pesos
  del modelo ni datos de usuarios).
"""
import os
import secrets

from fastapi import Header, HTTPException, status

HEADER_NAME = "X-Internal-Key"


async def require_internal_key(x_internal_key: str | None = Header(default=None, alias=HEADER_NAME)) -> None:
    expected = os.environ.get("AI_INTERNAL_KEY", "").strip()
    if not expected:
        return  # proteccion desactivada
    given = (x_internal_key or "").strip()
    if not secrets.compare_digest(given.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Clave interna invalida o ausente")
