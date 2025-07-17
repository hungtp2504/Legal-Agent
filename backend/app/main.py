import logging
import sys
from datetime import datetime

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from .api.v1 import chat
from .core.settings import settings


log_filename = datetime.now().strftime(f"backend_log_%Y-%m-%d.log")

settings.LOG_DIR.mkdir(exist_ok=True)
log_filepath = settings.LOG_DIR / log_filename

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

app = FastAPI(
    title="LegalAgent API",
    version="1.0",
    description="Backend service for the LegalAgent application",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])


@app.get("/health", status_code=200, tags=["Health"])
def health_check():
    return {"status": "ok"}
