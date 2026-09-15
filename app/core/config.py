import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Tactical Sim AI Engine"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    PORT: int = 8000
    SECRET_KEY: str = os.getenv("JWT_SECRET", "404E635266556A586E3272357538782F413F4428472B4B6250645367566B5970")

    class Config:
        case_sensitive = True

settings = Settings()