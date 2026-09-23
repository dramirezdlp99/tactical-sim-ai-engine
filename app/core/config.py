import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Tactical Sim AI Engine"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    PORT: int = 8000

    # SEGURIDAD: antes este valor de respaldo era EL MISMO que el JWT
    # secret comprometido del backend (expuesto en el historial de git
    # de un repo publico). Dos servicios distintos no deberian compartir
    # secreto -- si uno se compromete, el otro tambien queda afectado.
    #
    # Ahora:
    #  1. Se lee de una variable de entorno DISTINTA (AI_ENGINE_SECRET_KEY,
    #     no JWT_SECRET) para que nunca dependan del mismo valor.
    #  2. El valor de respaldo es uno NUEVO, generado con secrets.token_hex,
    #     que nunca ha existido en ningun commit anterior.
    #
    # PARA EL DESPLIEGUE: en el servidor donde corra este AI Engine
    # (Render, Azure, VPS, etc.), agrega la variable de entorno:
    #   AI_ENGINE_SECRET_KEY=<genera uno nuevo, distinto a este de respaldo>
    SECRET_KEY: str = os.getenv(
        "AI_ENGINE_SECRET_KEY",
        "2bd03ec9ba1d1c9f2e373d38e95f5778e692e552b3006f2e5b444b40ca431ca2",
    )

    class Config:
        case_sensitive = True


settings = Settings()