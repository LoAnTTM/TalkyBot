import threading
import time
from components.stt import SpeechToText
from components.brain import Chatbot
from components.logger import get_logger

class STTConversationThread(threading.Thread):
    def __init__(self, state_manager=None, response_callback=None):
        super().__init__()
        self.stt = SpeechToText()
        self.chatbot = Chatbot()
        self.state_manager = state_manager
        self.response_callback = response_callback
        self._stop_event = threading.Event()
        self.logger = get_logger("STT")

    def process_frame(self, frame):
        self.logger.debug("STTConversationThread received frame for processing")
        # Xử lý frame audio bằng STT
        text = self.stt.process_frame(frame)
        if text:
            self.logger.info(f"Final result: '{text}'")
            response = self.chatbot.get_response(text)
            self.logger.info(f"Bot response: {response}")
            if self.response_callback:
                self.response_callback(response)

    def run(self):
        self.logger.info("STT thread started, waiting for frames...")
        while not self._stop_event.is_set():
            time.sleep(0.1)  # Idle loop, chỉ dừng khi stop

    def stop(self):
        self._stop_event.set()