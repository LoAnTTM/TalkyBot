import os
import pygame
import time
from TTS.utils.manage import ModelManager
from TTS.api import TTS

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Configure the model name you want to use (using available models)
TTS_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"
VOCODER_MODEL_NAME = "vocoder_models/en/ljspeech/hifigan_v2"

# Model storage directory
MODEL_DIR = "models/tts"

# Create directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to download model if missing
def download_model_if_missing(model_name, target_dir):
    manager = ModelManager()
    model_path = os.path.join(target_dir, model_name.replace("/", "__"))
    if not os.path.exists(model_path):
        print(f"⬇️  Downloading model: {model_name}")
        # Fix: ModelManager.download_model() only takes model_name parameter
        manager.download_model(model_name)
        # The model will be downloaded to the default TTS cache directory
        return None  # We'll use TTS() with model_name instead
    else:
        print(f"✅ Model already exists: {model_name}")
        return model_path

class TextToSpeech:
    def __init__(self):
        """Initialize TTS with specified models"""
        print("🎤 Initializing Text-to-Speech...")
        try:
            # Try to download models if needed
            download_model_if_missing(TTS_MODEL_NAME, MODEL_DIR)
            download_model_if_missing(VOCODER_MODEL_NAME, MODEL_DIR)
            
            # Load model using model name (TTS will handle paths automatically)
            self.tts = TTS(
                model_name=TTS_MODEL_NAME,
                vocoder_name=VOCODER_MODEL_NAME,
                progress_bar=False
            )
            print("✅ TTS initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize TTS: {e}")
            print("Falling back to simpler model...")
            try:
                # Fallback to a simpler model
                self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
                print("✅ TTS initialized with fallback model!")
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                self.tts = None
    
    def speak(self, text, output_file="output.wav", play_audio=True):
        """Convert text to speech, save as file and optionally play it"""
        if not self.tts:
            print("❌ TTS not available")
            return False
            
        try:
            print(f"🗣️ Speaking: {text}")
            self.tts.tts_to_file(text=text, file_path=output_file)
            
            if play_audio:
                self.play_audio_file(output_file)
                
            return True
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return False
    
    def play_audio_file(self, file_path):
        """Play audio file using pygame"""
        try:
            print(f"🔊 Playing audio: {file_path}")
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            print("✅ Audio playback finished")
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
    
    def speak_and_play(self, text):
        """Convenience method to speak text and play it immediately"""
        return self.speak(text, "temp_output.wav", play_audio=True)
    
    def is_available(self):
        """Check if TTS is available"""
        return self.tts is not None

# For backward compatibility, create a global instance
def get_tts_instance():
    """Get a global TTS instance"""
    global _tts_instance
    if '_tts_instance' not in globals():
        _tts_instance = TextToSpeech()
    return _tts_instance
