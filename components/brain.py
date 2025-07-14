from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
import torch

class Chatbot:
    def __init__(self, model_name="facebook/blenderbot_small-90M", device=None):
        self.tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_name)
        self.model = BlenderbotSmallForConditionalGeneration.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.history = []

    def get_response(self, user_input):
        context = f"{user_input}".strip()  

        inputs = self.tokenizer(context, return_tensors="pt", padding="longest").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.history.append(user_input)
        self.history.append(response)
        return response.strip()

    def reset_conversation(self):
        self.history = []

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
