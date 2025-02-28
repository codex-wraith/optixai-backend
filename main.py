import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
from cipher_agent import CipherAgent
from datetime import datetime, UTC
from contextlib import asynccontextmanager
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from hypercorn.config import Config
from hypercorn.asyncio import serve
from perplexity_client import PerplexityClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        # Initialize basic services
        redis_url = os.environ.get('REDISCLOUD_URL')
        if not redis_url:
            logger.warning("REDISCLOUD_URL not found in environment")
            
        anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment")
            
        # Create a shared aiohttp session (persistent for external API calls)
        shared_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        
        # Initialize core components
        logger.info("Initializing CipherAgent...")
        app.state.agent = CipherAgent(
            perplexity_client=PerplexityClient(
                token=os.environ.get("PERPLEXITY_API_TOKEN"),
                session=shared_session
            )
        )

        yield

    finally:
        # Shutdown sequence
        logger.info("Starting shutdown sequence...")
        
        # Close the shared aiohttp session in the PerplexityClient
        if hasattr(app.state.agent, 'perplexity_client'):
            await app.state.agent.perplexity_client.close()
        
        logger.info("Shutdown sequence completed")

app = FastAPI(lifespan=lifespan)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

class Message(BaseModel):
    text: str
    platform: str = "telegram"
    user_id: str

@app.post("/chat")
async def chat(message: Message):
    try:
        if not app.state.agent:
            raise HTTPException(
                status_code=503,
                detail="Chat service not initialized. Please try again later."
            )
            
        # Validate Web requests include a user_id (session ID)
        if message.platform == "web" and not message.user_id:
            raise HTTPException(
                status_code=400,
                detail="Web platform requires a session ID"
            )
            
        response = await app.state.agent.respond(
            message.text, 
            platform=message.platform,
            user_id=message.user_id
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Matrix connection interrupted")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if not hasattr(app.state, 'agent'):
            raise HTTPException(status_code=503, detail="Services not initialized")
        return {
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "agent_status": "ready"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

port = int(os.environ.get("PORT", 8000))

config = Config()
config.bind = [f"0.0.0.0:{port}"]

async def run():
    """Run the web app with background services managed by the lifespan context"""
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(run())
