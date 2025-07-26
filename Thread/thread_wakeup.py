import threading
import time
from components.wakeword import WakeWordDetector
from components.logger import get_logger

class WakeWordThread(threading.Thread):
    def __init__(self, state_manager=None):
        super().__init__()
        self.daemon = True
        self.state_manager = state_manager
        self.detector = WakeWordDetector(state_manager=state_manager)
        self._stop_event = threading.Event()
        self.logger = get_logger("WAKE")
    
    def process_frame(self, frame):
        try:
            self.detector.process_frame(frame)
        except Exception as e:
            self.logger.error(f"Error in wake word process_frame: {e}")

    def run(self):
        self.logger.info("Wake Word Thread started - waiting for frames from VAD")
        while not self._stop_event.is_set():
            time.sleep(0.1)  # Idle loop, waiting for frames from VAD
        self.logger.info("Wake Word Thread stopped")
    
    def stop(self):
        """Stop the wake word thread"""
        self._stop_event.set()
        if hasattr(self.detector, 'cleanup'):
            self.detector.cleanup()
