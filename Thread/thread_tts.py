import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import threading
import queue
import time
from components.tts import TextToSpeech


class TTSThread(threading.Thread):
    def __init__(self, max_queue_size=10):
        super().__init__()
        self.tts = TextToSpeech()
        self.text_queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self.is_speaking = False

    def run(self):
        print("🔊 TTS Thread started")
        while not self._stop_event.is_set():
            try:
                # Wait for new text to speak, timeout to check stop event
                text = self.text_queue.get(timeout=0.5)
                if text:
                    self.is_speaking = True
                    print(f"🔊 Speaking: {text}")
                    self.tts.speak(text)
                    self.is_speaking = False
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ TTS Error: {e}")
                self.is_speaking = False

        print("🛑 TTS Thread stopped")

    def stop(self):
        self._stop_event.set()
        # Stop playback if currently running
        if hasattr(self.tts, 'stop'):
            self.tts.stop()

    def speak_text(self, text, priority=False):
        """Add text to queue for speech synthesis"""
        if not text or not text.strip():
            return False
            
        try:
            if priority:
                # Clear queue for priority messages
                while not self.text_queue.empty():
                    try:
                        self.text_queue.get_nowait()
                    except queue.Empty:
                        break
                # Stop current speech if speaking
                if self.is_speaking:
                    self.tts.stop()
            
            self.text_queue.put(text.strip(), timeout=0.1)
            return True
        except queue.Full:
            print("⚠️ TTS queue is full, dropping message")
            return False

    def get_queue_size(self):
        """Get current queue size"""
        return self.text_queue.qsize()

    def clear_queue(self):
        """Clear all pending messages"""
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                break


if __name__ == "__main__":
    tts_thread = TTSThread()
    tts_thread.start()

    print("🔊 TTS Thread test started!")
    print("=" * 50)

    try:
        while True:
            text = input("\n📝 Enter text: ").strip()
            if text.lower() == "exit":
                break
            elif text.lower() == "clear":
                tts_thread.clear_queue()
                print("🗑️ Queue cleared")
                continue
            elif text.lower() == "status":
                print(f"📊 Queue size: {tts_thread.get_queue_size()}")
                print(f"📊 Currently speaking: {tts_thread.is_speaking}")
                continue
            elif text:
                priority = text.endswith('!')
                if priority:
                    text = text[:-1].strip()
                
                if tts_thread.speak_text(text, priority=priority):
                    status = "Priority" if priority else "Normal"
                    print(f"✅ Added to queue ({status})")
                else:
                    print("❌ Failed to add to queue")
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        print("🛑 Stopping TTS thread...")
        tts_thread.stop()
        tts_thread.join()
        print("✅ TTS thread stopped successfully")
