import threading
from components.wakeword import WakeWordDetector

class WakeWordThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.detector = WakeWordDetector()

    def run(self):
        self.detector.start_listening()
