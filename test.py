import os
from audio.mic_stream import AudioStream 

from vosk import Model, KaldiRecognizer
import numpy as np
import json
import threading
import time

class SpeechToText:
    def __init__(self, model_path="models/vosk/vosk-model-small-en-us-0.15", samplerate=16000, 
                 on_partial_result=None, on_final_result=None):
        """
        Initialize STT object with callback functions.
        :param on_partial_result: Function called with partial results (str).
        :param on_final_result: Function called with final results (str).
        """
        if not os.path.exists(model_path):
            raise RuntimeError(f"❌ Vosk model not found at {model_path}. Please download from https://alphacephei.com/vosk/models")
        
        self.model = Model(model_path)
        self.samplerate = samplerate
        self.on_partial_result = on_partial_result
        self.on_final_result = on_final_result
        
        self._thread = None
        self._stop_event = threading.Event()

    def _to_int16(self, audio):
        if audio.dtype != np.int16:
            return (audio * 32768).astype(np.int16)
        return audio

    def _listen_loop(self, frame_duration_ms=250):
        """Listening loop running in a separate thread."""
        print("🎙️  Thread: Starting to listen...")
        rec = KaldiRecognizer(self.model, self.samplerate)
        audio_stream = AudioStream(samplerate=self.samplerate, channels=1, frame_duration_ms=frame_duration_ms)
        
        for frame in audio_stream.stream():
            # Check if there's a stop signal
            if self._stop_event.is_set():
                break

            frame = self._to_int16(frame)
            if rec.AcceptWaveform(frame.tobytes()):
                result_text = json.loads(rec.Result()).get("text", "").strip()
                if result_text and self.on_final_result:
                    self.on_final_result(result_text)
            else:
                partial_text = json.loads(rec.PartialResult()).get("partial", "").strip()
                if partial_text and self.on_partial_result:
                    self.on_partial_result(partial_text)
        
        print("🛑 Thread: Stopped listening.")

    def start(self):
        """Start the listening thread."""
        if self._thread and self._thread.is_alive():
            print("⚠️ Thread is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_loop)
        self._thread.daemon = True  # Thread will automatically close when main program ends
        self._thread.start()

    def stop(self):
        """Stop the listening thread."""
        if not self._thread or not self._thread.is_alive():
            print("ℹ️ Thread hasn't been started or has already stopped.")
            return
            
        self._stop_event.set()
        self._thread.join() # Wait for thread to completely finish
        self._thread = None

# --- Main Program ---
if __name__ == "__main__":
    
    # 1. Define callback functions to handle results
    def handle_partial_result(text):
        print(f"\r🔄 Partial: '{text}'", end='', flush=True)

    def handle_final_result(text):
        # Clear partial line and print final result
        print(f"\r{' '*80}\r✅ Final: '{text}'")

    # 2. Initialize SpeechToText object with callback functions
    stt = SpeechToText(
        on_partial_result=handle_partial_result,
        on_final_result=handle_final_result
    )

    # 3. Start listening thread
    stt.start()
    
    print("\n🚀 Main thread is running. Press Ctrl+C to exit.")
    print("You can speak anytime...")

    try:
        # Keep main thread alive so background thread can run
        while True:
            time.sleep(1)
            print(".", end="", flush=True) # Print to show main thread is not blocked
    except KeyboardInterrupt:
        print("\n⏳ Stopping program...")
    finally:
        # 4. Stop listening thread safely
        stt.stop()
        print("👋 Program has ended.")