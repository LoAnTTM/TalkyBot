import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import threading
import time
import json
import numpy as np
from vosk import Model, KaldiRecognizer

from components.stt import SpeechToText
from components.brain import Chatbot
from audio.mic_stream import AudioStream


class STTConversationThread(threading.Thread):
    def __init__(self, audio_queue=None, response_callback=None):
        super().__init__()
        self.chatbot = Chatbot()
        self.stt = SpeechToText()
        self.audio_queue = audio_queue
        self.response_callback = response_callback
        self._stop_event = threading.Event()
        
        # Initialize recognizer for continuous processing
        self.recognizer = KaldiRecognizer(self.stt.model, self.stt.samplerate)
        self.current_partial = ""

    def process_audio_frame(self, audio_frame):
        """Process a single audio frame for STT"""
        audio_int16 = self.stt._to_int16(audio_frame)
        
        if self.recognizer.AcceptWaveform(audio_int16.tobytes()):
            # Final result
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                self.on_final_result(text)
            self.current_partial = ""
        else:
            # Partial result
            partial = json.loads(self.recognizer.PartialResult())
            partial_text = partial.get("partial", "").strip()
            if partial_text and partial_text != self.current_partial:
                self.on_partial_result(partial_text)
                self.current_partial = partial_text

    def on_partial_result(self, text):
        """Handle partial STT results"""
        print(f"\r🔄 Partial: '{text}'", end='', flush=True)

    def on_final_result(self, text):
        """Handle final STT results and generate chatbot response"""
        print(f"\r{' '*80}\r✅ Final: '{text}'")
        
        # Send question to chatbot and get response
        try:
            response = self.chatbot.get_response(text)
            print(f"🤖 Bot: {response}")
            
            # Send response to callback (e.g., for TTS)
            if self.response_callback:
                self.response_callback(response)
                
        except Exception as e:
            print(f"❌ Error generating response: {e}")

    def run(self):
        print("🎙️ Starting STT + Chatbot thread...")
        
        if self.audio_queue:
            # Process audio from queue (when integrated with VAD)
            while not self._stop_event.is_set():
                try:
                    # Get audio frame from queue with timeout
                    audio_frame = self.audio_queue.get(timeout=0.1)
                    if audio_frame is not None:
                        self.process_audio_frame(audio_frame)
                except:
                    continue
        else:
            # Process audio directly from microphone
            audio_stream = AudioStream(samplerate=16000, channels=1, frame_duration_ms=250)
            
            try:
                for frame in audio_stream.stream():
                    if self._stop_event.is_set():
                        break
                    self.process_audio_frame(frame)
            except Exception as e:
                print(f"❌ Error in audio processing: {e}")
        
        print("🛑 STT + Chatbot thread stopped.")

    def stop(self):
        self._stop_event.set()


if __name__ == "__main__":
    def response_callback(response):
        print(f"📢 Response ready for TTS: {response}")
    
    # Test the STT conversation thread
    thread = STTConversationThread(response_callback=response_callback)
    thread.start()

    try:
        print("🎙️ STT + Chatbot thread test started!")
        print("Speak to test (Press Ctrl+C to stop)")
        print("=" * 50)
        
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping STT + Conversation thread...")
        thread.stop()
        thread.join()
        print("✅ Thread stopped successfully!")
