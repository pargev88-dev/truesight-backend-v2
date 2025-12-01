# app/core/config.py
import os
from functools import lru_cache

from pydantic import BaseSettings


class Settings(BaseSettings):
    ENV: str = os.getenv("TS_ENV", "development")
    VIDEO_MODEL_PATH: str = os.getenv(
        "TS_VIDEO_MODEL_PATH",
        "models/video/deepfake_model.pt"
    )
    ENABLE_GPT_EXPLANATION: bool = False  # set True if/when you add GPT back for explanations

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
