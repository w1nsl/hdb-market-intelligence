import os
import sys
import json
import re
from typing import Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate

development_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'development')
sys.path.insert(0, development_path)
from hdb_predictor import predict_hdb_price
from hdb_constraints import get_hdb_context
from hdb_database import HDBDatabase

load_dotenv()

# Configuration constants
CONFIG = {
    'MAX_CONVERSATION_HISTORY': 3,
    'DEFAULT_TEMPERATURE': 0.1,
    'CURRENT_YEAR': 2025,
    'NEW_BTO_LEASE': 99,
    'AVERAGE_VALUES': {
        'floor_area_sqm': {
            '2 ROOM': 45,
            '3 ROOM': 70,
            '4 ROOM': 90,
            '5 ROOM': 110
        },
        'storey_range': '07 TO 09',
        'flat_model': 'Model A'
    }
}

class HDBChatbotCore:
    """Core chatbot logic shared between console and service implementations"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        if self.verbose:
            print("🚀 Initializing HDB Chatbot...")
        
        self._initialize_llm()
        
        self._initialize_database()
        
        self.workflow = self._build_workflow()
        
        if self.verbose:
            print("✅ HDB Chatbot initialization complete!")
    
    def _initialize_llm(self):
        """Initialize and test LLM connection"""
        try:
            if self.verbose:
                print("🔄 Testing OpenAI connection...")
            
            self.llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=CONFIG['DEFAULT_TEMPERATURE']
            )
            
            test_response = self.llm.invoke([
                HumanMessage(content="Test connection. Reply with 'OK'.")
            ])
            
            if test_response.content.strip():
                if self.verbose:
                    print("✅ OpenAI connection successful")
            else:
                raise Exception("Empty response from OpenAI")
                
        except Exception as e:
            if self.verbose:
                print(f"❌ OpenAI connection failed: {e}")
            raise Exception(f"Cannot initialize chatbot without LLM connection: {e}")
    
    def _initialize_database(self):
        """Initialize and test database connection"""
        try:
            if self.verbose:
                print("🔄 Testing database connection...")
            
            if not os.getenv("NEON_DATABASE_URL"):
                raise Exception("NEON_DATABASE_URL environment variable not set")
            
            self.db = HDBDatabase()
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()
                
                if result and result[0] == 1:
                    if self.verbose:
                        print("✅ Database connection successful")
                    self.db_available = True
                else:
                    raise Exception("Database test query failed")
                    
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Database connection failed: {e}")
                print("📋 Continuing with prediction-only mode (data queries unavailable)")
            self.db = None
            self.db_available = False
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with nodes"""
        workflow = StateGraph(Dict[str, Any])
        
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("make_prediction", self.make_prediction)
        workflow.add_node("ask_clarification", self.ask_clarification)
        workflow.add_node("general_question_handler", self.general_question_handler)
        workflow.add_node("data_query_handler", self.data_query_handler)
        
        workflow.set_entry_point("analyze_query")
        
        workflow.add_conditional_edges(
            "analyze_query",
            self._decide_next_action,
            {
                "predict": "make_prediction",
                "clarify": "ask_clarification",
                "general": "general_question_handler",
                "data_query": "data_query_handler"
            }
        )
        
        workflow.add_edge("make_prediction", END)
        workflow.add_edge("ask_clarification", END) 
        workflow.add_edge("general_question_handler", END)
        workflow.add_edge("data_query_handler", END)
        
        return workflow.compile()
    
    def chat(self, user_input: str, chat_history: list = None, conversation_history: list = None) -> dict:
        """Main chat interface - uses LangGraph workflow
        
        Args:
            user_input: The latest user message
            chat_history: List of previous messages [{'role': 'user'/'assistant', 'content': '...'}, ...]
            conversation_history: Alternative format [{'user': '...', 'bot': '...'}, ...]
            
        Returns:
            dict: {'response': str, 'metadata': {'type': str, 'confidence': str, 'action': str}}
        """
        
        # Convert conversation_history format if provided
        if conversation_history is not None:
            chat_history = self._convert_conversation_to_chat_history(conversation_history)
        
        initial_state = {
            "user_input": user_input,
            "chat_history": chat_history or [],
            "user_intent": "",
            "extracted_info": {},
            "analysis": {},
            "response": "",
            "confidence": "",
            "action": "",
            "assumptions": [],
            "missing_info": [],
            "metadata": {}
        }
        
        try:
            result = self.workflow.invoke(initial_state)
            return {
                "response": result["response"],
                "metadata": {
                    "type": result.get("action", "unknown"),
                    "confidence": result.get("confidence", "medium"),
                    "action": result.get("action", "unknown"),
                    "assumptions": result.get("assumptions", []),
                    "missing_info": result.get("missing_info", [])
                }
            }
            
        except Exception as e:
            return {
                "response": f"I'm sorry, I encountered an error: {str(e)}\n\nPlease try rephrasing your question or type 'help' for examples.",
                "metadata": {
                    "type": "error",
                    "confidence": "low",
                    "action": "error"
                }
            }
    
    # ============== WORKFLOW NODES ==============
    
    def analyze_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node: Analyze user query and extract information"""
        
        user_input = state["user_input"]
        chat_history = state.get("chat_history", [])
                
        hdb_context = get_hdb_context()
        
        # Format chat history for context
        history_context = ""
        if chat_history:
            history_context = "\n\nCHAT HISTORY:\n"
            for i, msg in enumerate(chat_history[-6:]):  # Last 6 messages for context
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                history_context += f"{role.upper()}: {content}\n"
        
        analysis_prompt = PromptTemplate(
            template="""
Latest User Message: "{query}"
{history_context}
{hdb_context}

IMPORTANT: Analyze the LATEST user message in context of the chat history. Extract all information and apply intelligent inference:

INFERENCE RULES:
1. "new BTO" → remaining_lease: 99, lease_commence_date: 2025
2. "average" mentioned → infer typical values:
   - floor_area_sqm: 45 (2 ROOM), 70 (3 ROOM), 90 (4 ROOM), 110 (5 ROOM)
   - storey_range: "07 TO 09" (mid-level)
   - flat_model: "Model A" (most common)
3. Use chat history context to understand what information was already discussed
4. Extract ALL info from the latest message plus any relevant context from chat history

Apply inference rules automatically when keywords are detected.

Decide action:
1. PREDICT: ALL required parameters available (town, flat_type, floor_area_sqm, storey_range, flat_model, remaining_lease, lease_commence_date)
2. CLARIFY: ANY parameter missing after inference
3. DATA_QUERY: Questions about statistics, trends, comparisons (e.g., "most popular town", "average price", "which area has highest prices")
4. GENERAL: Non-prediction questions, greetings, help

Return JSON (use EXACT values from context lists):
{{
  "action": "predict/clarify/data_query/general",
  "extracted_info": {{
    "town": "extract from query or null",
    "flat_type": "extract from query or null",
    "floor_area_sqm": "extract/infer from query or null",
    "storey_range": "extract/infer from query or null",
    "flat_model": "extract/infer from query or null",
    "remaining_lease": "extract/infer from query or null",
    "lease_commence_date": "extract/infer from query or null"
  }},
  "assumptions": ["list inference assumptions made"],
  "missing_critical": ["list missing parameters"],
  "confidence": "high/medium/low"
}}""",
            input_variables=["query", "history_context", "hdb_context"]
        )
        
        try:
            
            messages = [
                SystemMessage(content="Senior HDB analyst for BTO development planning. Expert in Singapore's housing market. Analyze queries and return structured JSON using exact values from context."),
                HumanMessage(content=analysis_prompt.format(
                    query=user_input,
                    history_context=history_context,
                    hdb_context=hdb_context
                ))
            ]
            
            response = self.llm.invoke(messages)
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                state["analysis"] = analysis
                
                extracted_info = analysis.get("extracted_info", {})
                required_params = ["town", "flat_type", "floor_area_sqm", "storey_range", "flat_model", "remaining_lease", "lease_commence_date"]
                missing_params = [param for param in required_params if not extracted_info.get(param)]
                
                if missing_params and analysis.get("action") == "predict":
                    state["action"] = "clarify"
                    state["missing_info"] = missing_params
                else:
                    state["action"] = analysis.get("action", "clarify")
                    state["missing_info"] = analysis.get("missing_critical", [])
                
                state["extracted_info"] = extracted_info
                state["assumptions"] = analysis.get("assumptions", [])
                state["confidence"] = analysis.get("confidence", "medium")
            else:
                state["action"] = "clarify"
                state["missing_info"] = ["Could not understand query"]
                
        except Exception:
            state["action"] = "clarify" 
            state["missing_info"] = ["Error analyzing query"]
        
        return state
    
    def _decide_next_action(self, state: Dict[str, Any]) -> str:
        action = state.get("action", "clarify")
        
        # If database is not available, redirect data queries to general handler
        if action == "data_query" and not self.db_available:
            return "general"
        
        return action
    
    def make_prediction(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            info = state["extracted_info"]
            
            prediction = predict_hdb_price(
                town=info["town"],
                flat_type=info["flat_type"],
                floor_area_sqm=float(info["floor_area_sqm"]),
                storey_range=info["storey_range"],
                flat_model=info["flat_model"],
                remaining_lease=float(info["remaining_lease"]),
                lease_commence_date=int(info["lease_commence_date"])
            )
            
            response = self._format_prediction_response(state["user_input"], state["analysis"], prediction)
            state["response"] = response
            
        except Exception:
            state["response"] = "I encountered an issue making the prediction. Could you provide more specific details about the flat you're interested in?"
        
        return state
    
    def _format_prediction_response(self, query: str, analysis: Dict[str, Any], prediction: float) -> str:
        """LLM formats prediction response naturally"""
        
        formatting_prompt = PromptTemplate(
            template="""
Query: "{query}"
Info: {info}
Assumptions: {assumptions}
Price: S${price:,.0f}
Confidence: {confidence}

Create natural response with query acknowledgment, property summary, and clear price. Mention assumptions if confidence not high.
""",
            input_variables=["query", "info", "assumptions", "price", "confidence"]
        )
        
        try:
            messages = [
                SystemMessage(content="HDB assistant. Provide friendly price predictions."),
                HumanMessage(content=formatting_prompt.format(
                    query=query,
                    info=json.dumps(analysis.get("extracted_info", {}), indent=2),
                    assumptions=', '.join(analysis.get("assumptions", [])),
                    price=prediction,
                    confidence=analysis.get("confidence", "medium")
                ))
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception:
            return f"Based on your query, I estimate the HDB resale price at **S${prediction:,.0f}**. This is based on typical property characteristics for your requirements."
    
    def ask_clarification(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node: Ask clarifying questions"""
        
        user_input = state["user_input"]
        missing_info = state.get("missing_info", [])
        confidence = state.get("confidence", "low")
        chat_history = state.get("chat_history", [])
        
        # Check if we already have some info from chat history
        history_context = ""
        if chat_history:
            history_context = "\n\nPrevious conversation context:\n"
            for msg in chat_history[-4:]:  # Last 4 messages
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                history_context += f"{role.upper()}: {content}\n"
        
        clarification_prompt = PromptTemplate(
            template="""
Latest User Message: "{query}"
{history_context}
Missing Parameters: {missing}
Confidence: {confidence}

Create friendly response asking for the specific missing information. Consider what was already discussed in the chat history.
Be specific about what's needed for each parameter.

Parameter explanations:
- town: Which HDB town/area (e.g., Tampines, Jurong West)
- flat_type: Number of rooms (e.g., 2 ROOM, 3 ROOM, 4 ROOM, 5 ROOM, EXECUTIVE)
- floor_area_sqm: Floor area in square meters
- storey_range: Which floor range (e.g., 01 TO 03, 04 TO 06, 07 TO 09, etc.)
- flat_model: HDB flat model (e.g., Model A, Model C, Apartment, etc.)
- remaining_lease: Years of lease remaining
- lease_commence_date: Year when lease started
""",
            input_variables=["query", "history_context", "missing", "confidence"]
        )
        
        try:
            messages = [
                SystemMessage(content="HDB assistant. Ask clarifying questions conversationally."),
                HumanMessage(content=clarification_prompt.format(
                    query=user_input,
                    history_context=history_context,
                    missing=', '.join(missing_info),
                    confidence=confidence
                ))
            ]
            
            response = self.llm.invoke(messages)
            clarification_text = response.content
            
            state["response"] = clarification_text
            
        except Exception:
            state["response"] = """I'd love to help with an HDB price estimate! 

            To give you the most accurate prediction, could you tell me:
            • Which town or area are you looking at?
            • What type of flat (3-room, 4-room, 5-room, etc.)?

            Any additional details would help too!"""
                    
        return state
    
    def general_question_handler(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node: Handle general questions, help, greetings, and non-prediction queries using LLM"""
        
        user_input = state["user_input"]
        hdb_context = get_hdb_context()
        
        general_prompt = PromptTemplate(
            template="""
Query: "{query}"

HDB Context:
{hdb_context}

Respond as HDB Market Intelligence System. Be helpful and informative.
""",
            input_variables=["query", "hdb_context"]
        )
        
        try:
            messages = [
                SystemMessage(content="HDB Market Intelligence System. Expert on Singapore housing, BTO development, and all 26 towns. Provide helpful, accurate responses."),
                HumanMessage(content=general_prompt.format(query=user_input, hdb_context=hdb_context))
            ]
            
            response = self.llm.invoke(messages)
            state["response"] = response.content
            
        except Exception:
            state["response"] = """I'm your HDB Market Intelligence System! 

I can help you with:
• HDB price predictions and market analysis
• Information about Singapore's 26 towns and their characteristics  
• BTO development insights and planning
• Housing market trends and dynamics

Feel free to ask me anything about HDB or Singapore housing!"""
        
        return state
    
    def data_query_handler(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node: Handle data analysis queries using database"""
        
        user_input = state["user_input"]
        
        # Check if database is available
        if not self.db_available:
            state["response"] = "I apologize, but the data analysis feature is currently unavailable due to database connection issues. I can still help with HDB price predictions and general questions."
            return state
        
        try:
            # Query the database with natural language
            query_result = self.db.query_and_analyze(user_input)
            
            if query_result.get("error"):
                state["response"] = f"I encountered an issue with your data query: {query_result['error']}. Could you rephrase your question?"
                return state
            
            # Format the results with LLM
            response = self._format_data_query_response(user_input, query_result)
            state["response"] = response
            
        except Exception as e:
            state["response"] = f"I encountered an error while querying the data: {str(e)}. Please try rephrasing your question."
        
        return state
    
    def _format_data_query_response(self, query: str, query_result: Dict[str, Any]) -> str:
        """Format database query results into a natural response"""
        
        formatting_prompt = PromptTemplate(
            template="""
User Query: "{query}"
SQL Query: {sql}
Results: {results}
Total Records: {count}

Format these database results into a clear, informative response. 
- Start by showing the SQL query used (in a code block)
- Explain what the data shows
- Highlight key insights
- Use bullet points or tables if helpful
- Make it conversational and easy to understand
- If the results are numerical, provide context about what they mean
""",
            input_variables=["query", "sql", "results", "count"]
        )
        
        try:
            # Limit results display to prevent overly long responses
            display_results = query_result["results"][:10] if len(query_result["results"]) > 10 else query_result["results"]
            
            messages = [
                SystemMessage(content="HDB Data Analyst. Transform database query results into clear, insightful responses for users."),
                HumanMessage(content=formatting_prompt.format(
                    query=query,
                    sql=query_result["sql"],
                    results=display_results,
                    count=query_result["count"]
                ))
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception:
            # Fallback response if formatting fails
            results = query_result["results"]
            sql_query = query_result.get("sql", "SQL query not available")
            
            if not results:
                return f"**SQL Query Used:**\n```sql\n{sql_query}\n```\n\nI couldn't find any data matching your query. Please try a different question."
            
            response = f"**SQL Query Used:**\n```sql\n{sql_query}\n```\n\nBased on your query, I found {len(results)} results:\n\n"
            for i, result in enumerate(results[:5]):
                response += f"{i+1}. {str(result)}\n"
            
            if len(results) > 5:
                response += f"\n... and {len(results) - 5} more results."
            
            return response
    
    def _convert_conversation_to_chat_history(self, conversation_history: list) -> list:
        """Convert external conversation format to internal chat_history format
        
        Args:
            conversation_history: [{'user': '...', 'bot': '...'}, ...]
            
        Returns:
            list: [{'role': 'user'/'assistant', 'content': '...'}, ...]
        """
        chat_history = []
        
        for exchange in conversation_history:
            if 'user' in exchange:
                chat_history.append({
                    'role': 'user',
                    'content': exchange['user']
                })
            if 'bot' in exchange:
                chat_history.append({
                    'role': 'assistant', 
                    'content': exchange['bot']
                })
        
        return chat_history