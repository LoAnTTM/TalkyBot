import threading
import time
from components.wakeword import WakeWordDetector
from components.logger import get_logger

class WakeWordThread(threading.Thread):
    def __init__(self, state_manager=None):
        super().__init__()
        self.daemon = True
        self.state_manager = state_manager
        self.detector = WakeWordDetector()
        self._stop_event = threading.Event()
        self.logger = get_logger("WAKE")
        
        # Override the wake word callback to integrate with state manager
        self._setup_wake_word_callback()
    
    def _setup_wake_word_callback(self):
        """Setup wake word detection callback to trigger state transitions"""
        original_callback = self.detector._callback
        
        def state_aware_callback(indata, frames, time_info, status):
            # Only process wake word if system is in STANDBY
            if self.state_manager and self.state_manager.get_current_state().value != "STANDBY":
                return
            
            if status:
                self.logger.debug(f"Audio status: {status}")

            import numpy as np
            audio_int16 = (indata[:, 0] * 32767).astype(np.int16)

            try:
                scores = self.detector.model.predict(audio_int16)
                current_time = time.time()
                for wake_word, score in scores.items():
                    if (score > self.detector.sensitivity_threshold and 
                        (current_time - self.detector.last_trigger_time > self.detector.min_trigger_interval)):
                        
                        self.logger.info(f"Wake word '{wake_word}' detected! (score: {score:.3f})")
                        self.detector.last_trigger_time = current_time
                        
                        # Trigger state transition
                        if self.state_manager:
                            self.state_manager.wake_up(f"Wake word '{wake_word}' detected (score: {score:.3f})")
                        
            except Exception as e:
                self.logger.error(f"Error predicting wake word: {e}")
        
        # Replace the callback
        self.detector._callback = state_aware_callback

    def run(self):
        self.logger.info("Wake Word Thread started - Listening for 'Alexa'")
        try:
            self.detector.start_listening()
        except Exception as e:
            self.logger.error(f"Wake word detection error: {e}")
        finally:
            self.logger.info("Wake Word Thread stopped")
    
    def stop(self):
        """Stop the wake word thread"""
        self._stop_event.set()
        if hasattr(self.detector, 'cleanup'):
            self.detector.cleanup()
