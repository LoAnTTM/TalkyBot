from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Chatbot:
    def __init__(self, model_name="microsoft/DialoGPT-small", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.chat_history_ids = None

    def get_response(self, text):
        # Tokenize user input
        new_input_ids = self.tokenizer.encode(text + self.tokenizer.eos_token, return_tensors="pt").to(self.device)
        # Append to chat history
        if self.chat_history_ids is not None:
            bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids
        # Generate response
        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id
        )
        # Decode response
        response = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response.strip()
    
    def reset_conversation(self):
        """Reset chat history to start a new conversation"""
        self.chat_history_ids = None

# Test function
if __name__ == "__main__":
    print("🤖 Testing DialoGPT Chatbot...")
    
    try:
        # Initialize chatbot
        print("📥 Loading DialoGPT model...")
        chatbot = Chatbot()
        print("✅ DialoGPT model loaded successfully!")
        print(f"🖥️  Using device: {chatbot.device}")
        
        print("\n💬 Chat with the bot (type 'quit' to exit, 'reset' to restart conversation)")
        print("=" * 60)
        
        while True:
            # Get input from user
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'reset':
                chatbot.reset_conversation()
                print("🔄 Conversation reset!")
                continue
            elif not user_input:
                print("Please enter something!")
                continue
            
            try:
                # Generate response
                print("🤖 Bot: ", end="", flush=True)
                response = chatbot.get_response(user_input)
                print(response)
                
            except Exception as e:
                print(f"❌ Error generating response: {e}")
                
    except KeyboardInterrupt:
        print("\n👋 Chat ended by user.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure:")
        print("1. transformers library is installed: pip install transformers")
        print("2. torch is installed: pip install torch")
        print("3. Internet connection for downloading model (first run)")
        print("4. Sufficient memory for model loading") 