import threading
import time
import re

from components.stt import SpeechToText
from components.brain import Chatbot
from components.logger import get_logger

class STTConversationThread(threading.Thread):
    
    def __init__(self, state_manager=None, response_callback=None):
        super().__init__()
        self.stt = SpeechToText()
        self.chatbot = Chatbot()
        self.state_manager = state_manager
        self.response_callback = response_callback # Callback to send text to TTS
        self._stop_event = threading.Event()
        self.logger = get_logger("STT")

    def process_frame(self, frame):
        # Process audio frame with STT
        text = self.stt.process_frame(frame)
        if text:
            self.logger.info(f"Final result: '{text}'")
            # Check sleep keyword with separate function
            if self.contains_sleep_keyword(text):
                # self.logger.info("Sleep keyword detected, switching to STANDBY.")
                if self.state_manager and hasattr(self.state_manager, 'go_to_standby'):
                    self.state_manager.go_to_standby("Sleep keyword detected")
                    return
                return
            
            # Reset activity timeout
            if self.state_manager:
                self.state_manager.update_activity()

            response = self.chatbot.get_response(text)
            self.logger.info(f"Bot response: {response}")
            if self.response_callback:
                # Send response to TTS
                self.response_callback(response)

    def run(self):
        self.logger.info("STT thread started, waiting for frames...")
        while not self._stop_event.is_set():
            time.sleep(0.1)  # Idle loop, only stops when stop is called

    def stop(self):
        self._stop_event.set()

    def contains_sleep_keyword(self, text):
        sleep_keywords = [
            'bye', 'bye bye', 'goodbye', 'go to sleep', 'sleep',
            'stop listening', 'see you later', 'goodnight', 'good night',
            'shut down', 'power off', 'deactivate', 'turn off', 'night'
        ]
        text_lower = text.lower()
        # return any(keyword in text_lower for keyword in sleep_keywords)
        return any(re.search(rf'\b{re.escape(keyword)}\b', text_lower) for keyword in sleep_keywords)