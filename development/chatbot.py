import sys
import os
# Add service directory to path for imports
service_path = os.path.join(os.path.dirname(__file__), '..', 'service')
sys.path.insert(0, service_path)

from chatbot_core import HDBChatbotCore

class HDBChatbot(HDBChatbotCore):
    def __init__(self):
        # Initialize with verbose output for console interface
        super().__init__(verbose=True)
        self.conversation_history = []
    
    def _auto_combine_with_context(self, current_input: str) -> str:
        """Auto-combine context for console version"""
        # If no conversation history, return as-is
        if not self.conversation_history:
            return current_input
        
        last_bot_response = self.conversation_history[-1]['bot'].lower()
        
        # Check if last bot response was asking for clarification
        clarification_indicators = [
            "could you please", "could you tell me", "please specify", 
            "which town", "what type", "how many", "which floor",
            "missing", "need more information", "provide"
        ]
        
        is_clarification_request = any(indicator in last_bot_response for indicator in clarification_indicators)
        
        if is_clarification_request:
            # Get the original query from recent history
            original_query = ""
            for exchange in reversed(self.conversation_history[-3:]):
                user_msg = exchange['user'].lower()
                # Look for substantive queries (not just clarification responses)
                if (len(user_msg.split()) > 3 and 
                    any(keyword in user_msg for keyword in ['price', 'predict', 'bto', 'flat', 'hdb'])):
                    original_query = exchange['user']
                    break
            
            if original_query:
                return f"Previous: {original_query} Current: {current_input}"
        
        return current_input
    
    def chat(self, user_input: str) -> str:
        """Console chat interface with conversation history"""
        # Auto-detect if this is a response to a clarification
        combined_input = self._auto_combine_with_context(user_input)
        
        # Use parent's chat method with combined input
        response = super().chat(user_input, combined_input)
        
        # Store conversation history for console interface
        self.conversation_history.append({
            'user': user_input,
            'bot': response
        })
        
        return response

def start_chatbot():
    chatbot = HDBChatbot()
    print("""
    Hello! 👋 I'm your HDB Market Intelligence System.
    I specialize in analyzing Singapore's resale flat market to support BTO development planning and strategic decision-making.
    I can help with:
    • BTO Market Analysis "What's the market potential for 4-room BTO flats in Tampines?"
    • Comparative Town Analysis: "Compare resale prices between Jurong West and Sengkang for BTO planning"  
    • Development Insights: "Analyze market dynamics in Punggol for future BTO projects"
    • Price Predictions: "Predict resale values for new BTO archetypes in Woodlands"
    My analysis considers town characteristics, demographic trends, connectivity, and development potential to inform HDB's strategic planning.
    What market insights can I provide for you?
    """)
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue

            response = chatbot.chat(user_input)
            
            print(f"\n🤖 HDB Bot: {response}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using the HDB Price Predictor!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    start_chatbot()