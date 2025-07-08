import warnings
import threading
import numpy as np
import time

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
    def __init__(self, sensitivity=0.2, model_filter=None, confidence_smoothing=True):
        """
        Initialize wake word detector with OpenWakeWord

        Args:
            sensitivity (float): Detection sensitivity (0.0 to 1.0) - Lower = more sensitive
            model_filter (list): Only use specific models (e.g., ['alexa', 'hey_jarvis'])
            confidence_smoothing (bool): Apply confidence smoothing over multiple frames
        """
        self.model = None
        self.sensitivity = sensitivity
        self.wake_callback = None
        self.model_filter = model_filter
        self.confidence_smoothing = confidence_smoothing
        
        # For confidence smoothing
        self.confidence_history = {}
        self.history_length = 5  # Number of frames to average
        
        # Cooldown to prevent multiple detections
        self.last_detection_time = 0
        self.cooldown_seconds = 2.0  # Minimum time between detections

        self.setup_detector()

    def setup_detector(self):
        """Setup OpenWakeWord detection"""
        print("Setting up OpenWakeWord detection...")

        if not OPENWAKEWORD_AVAILABLE:
            print("\u274c OpenWakeWord not available")
            print("Install with: pip install openwakeword")
            self.model = None
            return False

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                openwakeword.utils.download_models()
                self.model = Model()

            print("\u2713 OpenWakeWord models loaded:")
            for model_name in self.model.models.keys():
                print(f"  - {model_name}")

            print("\u2713 Wake word detection ready!")
            return True

        except Exception as e:
            print(f"\u2717 Failed to initialize OpenWakeWord: {e}")
            self.model = None
            return False

    def set_wake_callback(self, callback_function):
        """Set the callback function to call when wake word is detected"""
        self.wake_callback = callback_function

    def _wake_word_detected(self, detected_model):
        """Internal callback when wake word is detected"""
        print(f"\n\ud83c\udfa4 Wake word '{detected_model}' detected!")
        if self.wake_callback:
            self.wake_callback()

    def detect_from_voice(self, audio_frame):
        """
        Call this method with audio frames that already passed VAD

        Args:
            audio_frame (np.ndarray): Audio data (float32 or int16)
        """
        if not self.model:
            return

        try:
            # Ensure audio_frame is numpy array
            if not isinstance(audio_frame, np.ndarray):
                audio_frame = np.array(audio_frame)
            
            # Ensure 1D array
            if audio_frame.ndim > 1:
                audio_frame = audio_frame.flatten()
            
            # Clean audio data - remove any invalid values
            audio_frame = np.nan_to_num(audio_frame, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Convert to int16 if needed
            if audio_frame.dtype == np.float32:
                # Clip values to valid range
                audio_frame = np.clip(audio_frame, -1.0, 1.0)
                audio_frame = (audio_frame * 32767).astype(np.int16)
            elif audio_frame.dtype != np.int16:
                audio_frame = audio_frame.astype(np.int16)

            predictions = self.model.predict(audio_frame)

            # Apply confidence smoothing if enabled
            if self.confidence_smoothing:
                predictions = self._smooth_predictions(predictions)
            
            # Filter models if specified
            if self.model_filter:
                predictions = {k: v for k, v in predictions.items() if k in self.model_filter}

            # Check for detection with cooldown
            current_time = time.time()
            if current_time - self.last_detection_time < self.cooldown_seconds:
                return  # Still in cooldown period

            for model_name, confidence in predictions.items():
                if confidence > self.sensitivity:
                    print(f"\ud83c\udfaf Detection: {model_name} (confidence: {confidence:.3f})")
                    self.last_detection_time = current_time
                    threading.Thread(
                        target=self._wake_word_detected,
                        args=(model_name,),
                        daemon=True
                    ).start()
                    break

        except Exception as e:
            # More detailed error logging
            error_msg = str(e)
            if "surrogates not allowed" in error_msg:
                print(f"⚠️ Audio encoding error - skipping frame (shape: {audio_frame.shape if 'audio_frame' in locals() else 'unknown'})")
            else:
                print(f"Wake word detection error: {error_msg}")

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
                print(f"\ud83d\udca4 Sleep keyword detected: '{keyword}'")
                return True
        return False

    def adjust_sensitivity(self, new_sensitivity):
        """
        Adjust wake word detection sensitivity

        Args:
            new_sensitivity (float): New sensitivity value (0.0 to 1.0)
        """
        self.sensitivity = max(0.0, min(1.0, new_sensitivity))
        print(f"\ud83c\udf9b\ufe0f Wake word sensitivity adjusted to {self.sensitivity}")

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
        return info

    def reset_models(self):
        """Reset the model buffers (useful after long activation periods)"""
        if self.model:
            try:
                self.model.reset()
                print("\ud83d\udd04 Model buffers reset")
            except Exception as e:
                print(f"Error resetting models: {e}")

    def configure_sensitivity(self, sensitivity=None, model_filter=None, cooldown=None, smoothing=None):
        """
        Configure wake word detection parameters
        
        Args:
            sensitivity (float): Detection threshold (lower = more sensitive)
            model_filter (list): Only detect specific models
            cooldown (float): Seconds between detections
            smoothing (bool): Enable confidence smoothing
        """
        if sensitivity is not None:
            self.sensitivity = max(0.0, min(1.0, sensitivity))
            print(f"🎛️ Sensitivity set to {self.sensitivity}")
            
        if model_filter is not None:
            self.model_filter = model_filter
            print(f"🎯 Model filter set to {self.model_filter}")
            
        if cooldown is not None:
            self.cooldown_seconds = cooldown
            print(f"⏰ Cooldown set to {self.cooldown_seconds}s")
            
        if smoothing is not None:
            self.confidence_smoothing = smoothing
            if not smoothing:
                self.confidence_history = {}  # Reset history
            print(f"📊 Confidence smoothing {'enabled' if smoothing else 'disabled'}")

    def get_detection_stats(self):
        """Get current detection configuration"""
        return {
            'sensitivity': self.sensitivity,
            'model_filter': self.model_filter,
            'cooldown_seconds': self.cooldown_seconds,
            'confidence_smoothing': self.confidence_smoothing,
            'available_models': self.get_available_models()
        }

    def _smooth_predictions(self, predictions):
        """Apply confidence smoothing over multiple frames"""
        smoothed = {}
        
        for model_name, confidence in predictions.items():
            # Initialize history if not exists
            if model_name not in self.confidence_history:
                self.confidence_history[model_name] = []
            
            # Add current confidence
            self.confidence_history[model_name].append(confidence)
            
            # Keep only recent history
            if len(self.confidence_history[model_name]) > self.history_length:
                self.confidence_history[model_name] = self.confidence_history[model_name][-self.history_length:]
            
            # Calculate smoothed confidence (moving average)
            smoothed[model_name] = sum(self.confidence_history[model_name]) / len(self.confidence_history[model_name])
        
        return smoothed
