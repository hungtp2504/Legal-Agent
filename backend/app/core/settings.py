import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT_DIR", Path.cwd()))

load_dotenv(PROJECT_ROOT / ".env")

class Settings(BaseSettings):
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    LANGFUSE_SECRET_KEY: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY: Optional[str] = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_HOST: Optional[str] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    LLM_MODEL_NAME: str = "gpt-4o-mini"
    EMBEDDING_MODEL_NAME: str = "bkai-foundation-models/vietnamese-bi-encoder"

    DATA_DIR: Path = PROJECT_ROOT / "data"
    LOG_DIR: Path = PROJECT_ROOT / "logs"
    CHROMA_PERSIST_PATH: Path = DATA_DIR / "chroma_db"
    PARSED_JSON_DIR: Path = DATA_DIR / "parsed_json_output"

    CHROMA_COLLECTION_NAME: str = "bo_phap_dien_viet_nam"

    def __init__(self, **values):
        super().__init__(**values)
        self.LOG_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)
        self.PARSED_JSON_DIR.mkdir(exist_ok=True)
        self.CHROMA_PERSIST_PATH.mkdir(exist_ok=True)

settings = Settings()
