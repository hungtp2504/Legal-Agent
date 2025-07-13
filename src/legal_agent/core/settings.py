import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    GEMINI_API_KEY: Optional[str] = None

    LLM_MODEL_NAME: str = "gemini-1.5-flash-latest"
    EMBEDDING_MODEL_NAME: str = "bkai-foundation-models/vietnamese-bi-encoder"

    DATA_DIR: Path = BASE_DIR / "data"
    LOG_DIR: Path = BASE_DIR / "logs"

    RAW_HTML_DIR: Path = DATA_DIR / "bo_phap_dien_html"
    RAW_TEXT_DIR: Path = DATA_DIR / "bo_phap_dien_raw_text"
    PARSED_JSON_DIR: Path = DATA_DIR / "parsed_json_output"
    CHROMA_PERSIST_PATH: Path = DATA_DIR / "chroma_db"

    CHROMA_COLLECTION_NAME: str = "bo_phap_dien_viet_nam"

    def __init__(self, **values):
        super().__init__(**values)

        self.LOG_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)
        self.RAW_HTML_DIR.mkdir(exist_ok=True)
        self.RAW_TEXT_DIR.mkdir(exist_ok=True)
        self.PARSED_JSON_DIR.mkdir(exist_ok=True)
        self.CHROMA_PERSIST_PATH.mkdir(exist_ok=True)


settings = Settings()

