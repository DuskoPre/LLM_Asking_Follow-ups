from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from ..core.config import Config
from ..core.system import AskFollowupsSystem

# Initialize FastAPI app
app = FastAPI(
    title="Ask Follow-ups API",
    description="Intelligent Q&A system with automatic clarification",
    version="1.0.0"
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    conversation_history: Optional[List[Dict]] = []

class ClarificationRequest(BaseModel):
    original_query: str
    clarification_response: str
    conversation_history: Optional[List[Dict]] = []

class QueryResponse(BaseModel):
    action: str
    answer: Optional[str] = None
    clarification_questions: Optional[List[str]] = []
    confidence_scores: Dict[str, float]
    retrieved_docs_count: int

# Initialize system
config = Config(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
)
system = AskFollowupsSystem(config)

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query"""
    try:
        result = await system.process_query(
            request.query, 
            request.conversation_history
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clarify", response_model=QueryResponse)
async def process_clarification(request: ClarificationRequest):
    """Process clarification response"""
    try:
        result = await system.process_clarification_response(
            request.original_query,
            request.clarification_response,
            request.conversation_history
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Ask Follow-ups API", "docs": "/docs"}
