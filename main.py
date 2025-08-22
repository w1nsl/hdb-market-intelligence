from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import ChatRequest, ChatResponse, HealthResponse
from service.chatbot_core import HDBChatbotCore
import uuid
from typing import Dict, List
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

# In-memory session storage
sessions: Dict[str, Dict] = {}
SESSION_TIMEOUT = timedelta(hours=2)  # Sessions expire after 2 hours

# Global chatbot service instance
chatbot_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global chatbot_service
    
    try:
        # Initialize chatbot service on startup
        print("🚀 Starting HDB Chatbot API Service...")
        chatbot_service = HDBChatbotCore(verbose=False)
        print("✅ HDB Chatbot Service initialized successfully!")
        
        # Start session cleanup task
        cleanup_task = asyncio.create_task(cleanup_sessions())
        
        yield
        
        # Cleanup on shutdown
        cleanup_task.cancel()
        print("👋 HDB Chatbot API Service shutting down...")
        
    except Exception as e:
        print(f"❌ Failed to initialize HDB Chatbot Service: {e}")
        raise

# Create FastAPI app with lifespan management
app = FastAPI(
    title="HDB Market Intelligence API",
    description="AI-powered chatbot API for Singapore HDB resale market analysis and BTO planning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def cleanup_sessions():
    """Background task to clean up expired sessions"""
    while True:
        try:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session_data in sessions.items():
                if current_time - session_data["last_activity"] > SESSION_TIMEOUT:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del sessions[session_id]
                print(f"🧹 Cleaned up expired session: {session_id}")
            
            # Run cleanup every 30 minutes
            await asyncio.sleep(1800)
            
        except Exception as e:
            print(f"❌ Error in session cleanup: {e}")
            await asyncio.sleep(1800)

def get_or_create_session(session_id: str = None) -> str:
    """Get existing session or create new one"""
    if session_id and session_id in sessions:
        # Update last activity
        sessions[session_id]["last_activity"] = datetime.now()
        return session_id
    
    # Create new session
    new_session_id = str(uuid.uuid4())
    sessions[new_session_id] = {
        "conversation_history": [],
        "created_at": datetime.now(),
        "last_activity": datetime.now()
    }
    return new_session_id

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HDB Market Intelligence API",
        "version": "1.0.0",
        "description": "AI-powered chatbot for Singapore HDB market analysis",
        "endpoints": {
            "chat": "POST /chat - Main chatbot interaction",
            "health": "GET /health - Health check",
            "docs": "GET /docs - API documentation"
        },
        "features": [
            "HDB price predictions",
            "Market data queries", 
            "BTO planning insights",
            "Natural language interaction"
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for interacting with HDB chatbot"""
    
    if chatbot_service is None:
        raise HTTPException(status_code=503, detail="Chatbot service not available")
    
    try:
        # Get or create session
        session_id = get_or_create_session(request.session_id)
        session_data = sessions[session_id]
        
        # Get chatbot response
        bot_response = chatbot_service.chat(
            user_input=request.message,
            conversation_history=session_data["conversation_history"]
        )
        response_text = bot_response["response"]
        
        # Update conversation history
        session_data["conversation_history"].append({
            "user": request.message,
            "bot": response_text
        })
        
        # Keep only recent history to prevent memory issues
        if len(session_data["conversation_history"]) > 10:
            session_data["conversation_history"] = session_data["conversation_history"][-10:]
        
        session_data["last_activity"] = datetime.now()
        
        return ChatResponse(
            response=response_text,
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    
    if chatbot_service is None:
        raise HTTPException(status_code=503, detail="Chatbot service not initialized")
    
    return HealthResponse(
        status="healthy",
        database_available=chatbot_service.db_available
    )

@app.get("/sessions")
async def get_sessions():
    """Debug endpoint to view active sessions (remove in production)"""
    return {
        "active_sessions": len(sessions),
        "sessions": {
            session_id: {
                "created_at": data["created_at"].isoformat(),
                "last_activity": data["last_activity"].isoformat(),
                "conversation_length": len(data["conversation_history"])
            }
            for session_id, data in sessions.items()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)