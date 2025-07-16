import logging
import json
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ...core.settings import settings
from ...legal_agent.agent.agent_runner import get_agent_runner

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    query: str

@router.post("/chat/stream")
async def stream_chat(request: Request, chat_request: ChatRequest):

    if not chat_request.query:
        raise HTTPException(status_code=400, detail="Query is required.")

    try:
        
        agent_runner = get_agent_runner()
    except Exception as e:
        logger.error(f"Failed to initialize Agent Runner: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent initialization failed: {e}")

    async def event_generator():
        try:
            async for event in agent_runner.stream_run(chat_request.query):
                if await request.is_disconnected():
                    logger.warning("Client disconnected, stopping stream.")
                    break
                
                yield {"data": json.dumps(event)}

        except Exception as e:
            logger.error(f"Error during agent execution stream: {e}", exc_info=True)
            error_event = {"type": "error", "data": str(e)}
            yield {"data": json.dumps(error_event)}

    return EventSourceResponse(event_generator())