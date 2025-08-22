from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User message to send to chatbot")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Chatbot response")
    session_id: str = Field(..., description="Session ID for this conversation")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    database_available: bool = Field(..., description="Whether database is available")