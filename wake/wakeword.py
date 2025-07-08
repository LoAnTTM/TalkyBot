import os
import threading
import time
import warnings
import numpy as np
import sounddevice as sd
from collections import deque

# Suppress warnings before importing openwakeword to avoid tflite runtime warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import openwakeword

try:
    from openwakeword.model import Model
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    print("⚠️ OpenWakeWord not available - install with: pip install openwakeword")

class WakeWordDetector:
    def __init__(self, sensitivity=0.5):
        """
        Initialize wake word detector with OpenWakeWord
        
        Args:
            sensitivity (float): Detection sensitivity (0.0 to 1.0)
        """
        self.model = None
        self.sensitivity = sensitivity
        self.is_listening = False
        self.wake_callback = None
        self.audio_stream = None
        
        # Audio settings (must match OpenWakeWord requirements)
        self.CHUNK = 1280  # 80ms at 16kHz (required by OpenWakeWord)
        self.CHANNELS = 1
        self.RATE = 16000
        
        # Audio buffer for continuous processing
        self.audio_buffer = deque(maxlen=self.CHUNK * 10)
        
        self.setup_detector()
    
    def setup_detector(self):
        """Setup OpenWakeWord detection"""
        print("Setting up OpenWakeWord detection...")
        
        if not OPENWAKEWORD_AVAILABLE:
            print("❌ OpenWakeWord not available")
            print("Install with: pip install openwakeword")
            self.model = None
            return False
        
        try:
            # Download models if not already present (one-time download)
            print("Checking for pre-trained models...")
            
            # Suppress warnings during model download and initialization
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                openwakeword.utils.download_models()
                
                # Initialize OpenWakeWord model with default models
                # This loads all pre-trained models by default
                self.model = Model()
            
            print("✓ OpenWakeWord models loaded:")
            available_models = list(self.model.models.keys())
            for model_name in available_models:
                print(f"  - {model_name}")
            
            print("✓ Wake word detection ready!")
            print("Available wake phrases:")
            for model_name in available_models:
                # Convert model names to readable phrases
                if 'alexa' in model_name.lower():
                    print("  - 'Alexa'")
                elif 'jarvis' in model_name.lower():
                    print("  - 'Hey Jarvis'")
                elif 'mycroft' in model_name.lower():
                    print("  - 'Hey Mycroft'")
                elif 'timer' in model_name.lower():
                    print("  - 'Timer'")
                elif 'weather' in model_name.lower():
                    print("  - 'Weather'")
                else:
                    # Generic conversion for other models
                    readable_name = model_name.replace('_', ' ').replace('.tflite', '').replace('.onnx', '').title()
                    print(f"  - '{readable_name}'")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize OpenWakeWord: {e}")
            self.model = None
            return False
    
    def set_wake_callback(self, callback_function):
        """Set the callback function to call when wake word is detected"""
        self.wake_callback = callback_function
    
    def _wake_word_detected(self, detected_model):
        """Internal callback when wake word is detected"""
        print(f"\n🎤 Wake word '{detected_model}' detected!")
        if self.wake_callback:
            self.wake_callback()
    
    def start_listening(self):
        """Start listening for wake words"""
        if not self.model:
            print("❌ No wake word model available")
            return False
        
        print("\n🔊 Listening for wake words...")
        print("Say any of the available wake phrases to activate")
        
        try:
            self.is_listening = True
            self._start_audio_stream()
            return True
        except Exception as e:
            print(f"Error starting wake word detection: {e}")
            return False
    
    def _start_audio_stream(self):
        """Start continuous audio stream for wake word detection"""
        try:
            print("🎤 Starting audio stream for wake word detection")
            
            # Start sounddevice stream with callback
            self.audio_stream = sd.InputStream(
                channels=self.CHANNELS,
                samplerate=self.RATE,
                blocksize=self.CHUNK,
                dtype=np.int16,
                callback=self._audio_callback
            )
            
            self.audio_stream.start()
            print("🎤 Audio stream started for wake word detection")
            
            # Keep main thread alive while listening
            while self.is_listening:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Audio stream error: {e}")
            self.stop_listening()
    
    def _audio_callback(self, indata, frames, time, status):
        """Sounddevice callback function for real-time audio processing"""
        if not self.is_listening:
            return
        
        if status:
            print(f"Audio callback status: {status}")
        
        try:
            # Convert audio data to int16 format as required by OpenWakeWord
            audio_frame = indata[:, 0]  # Get first channel
            if audio_frame.dtype == np.float32:
                # Convert from float32 [-1, 1] to int16 [-32768, 32767]
                audio_frame = (audio_frame * 32767).astype(np.int16)
            
            # Get predictions from OpenWakeWord
            predictions = self.model.predict(audio_frame)
            
            # Check each model's prediction against sensitivity threshold
            for model_name, confidence in predictions.items():
                if confidence > self.sensitivity:
                    print(f"🎯 Detection: {model_name} (confidence: {confidence:.3f})")
                    
                    # Use threading to avoid blocking the audio callback
                    detection_thread = threading.Thread(
                        target=self._wake_word_detected, 
                        args=(model_name,)
                    )
                    detection_thread.daemon = True
                    detection_thread.start()
                    
                    # Add small delay to prevent multiple rapid detections
                    time.sleep(0.5)
                    break
            
        except Exception as e:
            print(f"Wake word detection error: {e}")
    
    def stop_listening(self):
        """Stop listening for wake words"""
        self.is_listening = False
        
        try:
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
        except Exception as e:
            print(f"Error stopping audio: {e}")
                
        print("🔇 Wake word detection stopped")
    
    def stop_detection(self):
        """Stop wake word detection (alias for stop_listening)"""
        self.stop_listening()
    
    def is_available(self):
        """Check if wake word detection is available"""
        return self.model is not None
    
    def check_sleep_keywords(self, text):
        """
        Check if text contains sleep keywords
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if sleep keyword found
        """
        sleep_keywords = [
            'bye bye', 'goodbye', 'go to sleep', 'sleep', 
            'stop listening', 'see you later', 'goodnight',
            'shut down', 'power off', 'deactivate', 'turn off'
        ]
        
        text_lower = text.lower().strip()
        for keyword in sleep_keywords:
            if keyword in text_lower:
                print(f"💤 Sleep keyword detected: '{keyword}'")
                return True
        return False
    
    def adjust_sensitivity(self, new_sensitivity):
        """
        Adjust wake word detection sensitivity
        
        Args:
            new_sensitivity (float): New sensitivity value (0.0 to 1.0)
        """
        self.sensitivity = max(0.0, min(1.0, new_sensitivity))
        print(f"🎛️ Wake word sensitivity adjusted to {self.sensitivity}")
    
    def get_available_models(self):
        """Get list of available wake word models"""
        if not self.model:
            return []
        return list(self.model.models.keys())
    
    def get_model_info(self):
        """Get information about loaded models"""
        if not self.model:
            return "No models loaded"
        
        info = "Loaded wake word models:\n"
        for model_name in self.model.models.keys():
            info += f"  - {model_name}\n"
        
        info += f"\nCurrent sensitivity: {self.sensitivity}\n"
        info += f"Audio settings: {self.RATE}Hz, {self.CHANNELS} channel, {self.CHUNK} samples/chunk\n"
        
        return info
    
    def reset_models(self):
        """Reset the model buffers (useful after long activation periods)"""
        if self.model:
            try:
                self.model.reset()
                print("🔄 Model buffers reset")
            except Exception as e:
                print(f"Error resetting models: {e}")
    
    def test_detection(self):
        """Test wake word detection with current settings"""
        if not self.model:
            print("❌ No model available for testing")
            return False
        
        print("🧪 Testing wake word detection...")
        print("Say one of the available wake words within the next 10 seconds...")
        
        # Set up a temporary callback for testing
        original_callback = self.wake_callback
        test_detected = [False]
        
        def test_callback():
            test_detected[0] = True
            print("✅ Wake word test successful!")
        
        self.wake_callback = test_callback
        
        # Listen for 10 seconds
        test_start = time.time()
        temp_listening = self.is_listening
        
        if not self.is_listening:
            self.start_listening()
        
        while time.time() - test_start < 10 and not test_detected[0]:
            time.sleep(0.1)
        
        # Restore original callback
        self.wake_callback = original_callback
        
        if not temp_listening:
            self.stop_listening()
        
        if test_detected[0]:
            print("🎉 Wake word detection test passed!")
            return True
        else:
            print("⚠️ No wake word detected during test period")
            return False