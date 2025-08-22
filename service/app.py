from flask import Flask, render_template, request, jsonify
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from chatbot_core import HDBChatbotCore

app = Flask(__name__)

# Initialize chatbot service
try:
    chatbot_service = HDBChatbotCore(verbose=False)
    print("✅ HDB Chatbot Service initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize chatbot service: {e}")
    chatbot_service = None

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        if not chatbot_service:
            return jsonify({
                'error': 'Chatbot service not available',
                'response': 'Sorry, the chatbot service is currently unavailable. Please try again later.'
            }), 503
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Invalid request',
                'response': 'Please provide a message.'
            }), 400
        
        user_message = data['message'].strip()
        conversation_history = data.get('conversation_history', [])
        
        if not user_message:
            return jsonify({
                'error': 'Empty message',
                'response': 'Please enter a message.'
            }), 400
        
        # Get response from chatbot service
        bot_response = chatbot_service.chat(user_message, conversation_history=conversation_history)
        
        return jsonify({
            'response': bot_response['response'],
            'metadata': bot_response['metadata'],
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'response': f'Sorry, I encountered an error: {str(e)}. Please try again.'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'chatbot_service': 'available' if chatbot_service else 'unavailable'
    }
    return jsonify(status)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting HDB Market Intelligence Web Service...")
    print("📝 Make sure you have set your environment variables:")
    print("   - OPENAI_API_KEY")
    print("   - NEON_DATABASE_URL (optional)")
    print()
    
    # Check if running in development mode
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', 5001))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode
    )