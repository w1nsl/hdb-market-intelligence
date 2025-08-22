# HDB Market Intelligence System

AI-powered chatbot system for Singapore's HDB resale market analysis and BTO development planning.

## Features

- **Price Predictions** - ML-based HDB resale price forecasting
- **Data Queries** - Natural language SQL queries on real HDB transaction data
- **Market Analysis** - Insights and comparisons across Singapore's 26 towns  
- **Web Interface** - Simple chat interface for easy interaction
- **REST API** - JSON endpoints for integration

## High-Level Architecture

The HDB Market Intelligence System is built as a conversational AI that combines machine learning predictions with real-time data queries. Here's how it works:

### Core Components

1. **AI Chatbot Engine** (`chatbot_core.py`)
   - Uses LangGraph workflow for intelligent conversation routing
   - Analyzes user intent and extracts property information
   - Routes to appropriate handlers: prediction, data query, or clarification
   - Maintains chat history context for natural conversations

2. **ML Price Predictor** (`hdb_predictor.py`)
   - Trained scikit-learn model on historical HDB transaction data
   - Predicts resale prices based on 7 key factors: town, flat type, floor area, storey, model, lease info
   - Returns confidence scores and assumptions

3. **Database Query Engine** (`hdb_database.py`)
   - Natural language to SQL conversion for market data analysis
   - Real-time queries on HDB transaction database
   - Supports statistical analysis and trend comparisons

4. **Web Service**
   - **Flask Web App** (`service/app.py`) - Simple chat interface with REST API

### Conversation Flow

```
User Query → Intent Analysis → Route to Handler → Generate Response
     ↓              ↓              ↓                    ↓
 "4-room in     Extract info:   Prediction      "Based on market data,
  Tampines"     town, type,     Handler         price is $750,000"
                etc.
```

### Project Structure

```
GovTech Project/
├── development/              # Core ML and data components
│   ├── chatbot.py           # Console interface for testing
│   ├── hdb_predictor.py     # ML price prediction engine
│   ├── hdb_database.py      # Natural language SQL queries
│   ├── hdb_constraints.py   # Data validation and context
│   └── hdb_price_model/     # Trained ML models and encoders
├── service/                 # Web service layer
│   ├── app.py              # Flask web application
│   ├── chatbot_core.py     # Main chatbot engine
│   └── templates/          # Web interface (HTML/CSS/JS)
└── requirements.txt        # Python dependencies
```

## Quick Start

1. **Setup**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Model Setup**
   ```bash
   cd development
   python hdb_predictor.py  # This will train and save the model
   ```
   
   📦 **Model Note**: The trained model file (1.6GB) is excluded from this repository due to GitHub's size limits. The setup command above will generate it locally using the included HDB transaction data.

3. **Ready to Use**
   The `.env` file is included with shared development credentials for easy setup!

## Usage

### 1. Web Interface (Flask)
```bash
cd service
python app.py
# Visit http://localhost:5001 for web interface
# API available at http://localhost:5001/chat
```

### 2. Console Interface (Development)
```bash
cd development  
python chatbot.py
```

### Example Conversations

**Price Predictions:**
- "What's the price of a 4-room flat in Tampines?"
- "New BTO in Jurong West, 3-room, average floor area"
- "5-room executive in Bishan, 15 years old"

**Data Analysis:**
- "Which town has the highest average prices?"
- "Compare Jurong West vs Sengkang transaction volumes"
- "Show me the most expensive transactions this year"

**General Questions:**
- "What factors affect HDB prices?"
- "Tell me about Tampines as a town"
- "How does lease remaining affect price?"

## API Reference

### Flask Web App (`service/app.py`)
- `GET /` - Web chat interface
- `POST /chat` - Chat API
  ```json
  {
    "message": "4-room flat in Tampines",
    "conversation_history": [{"user": "...", "bot": "..."}]
  }
  ```
- `GET /health` - Service health check

### Response Format
```json
{
  "response": "Based on market data, the estimated price is $750,000",
  "metadata": {
    "type": "predict|clarify|general|data_query",
    "confidence": "high|medium|low",
    "assumptions": ["Assumed Model A flat", "..."]
  },
  "status": "success"
}
```

## Technical Stack

- **ML**: scikit-learn, pandas
- **AI**: LangChain, LangGraph, OpenAI 
- **Web**: Flask, HTML/CSS/JavaScript  
- **Database**: PostgreSQL 

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key (included) |
| `NEON_DATABASE_URL` | Yes | PostgreSQL connection string (included) |
