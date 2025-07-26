import threading
import time
import json
from vosk import Model, KaldiRecognizer

from components.stt import SpeechToText
from components.brain import Chatbot
from components.logger import get_logger
from components.state_manager import SystemState


class STTConversationThread(threading.Thread):
    def __init__(self, state_manager=None, audio_queue=None, response_callback=None):
        super().__init__()
        self.chatbot = Chatbot()
        self.stt = SpeechToText()
        self.state_manager = state_manager
        self.audio_queue = audio_queue
        self.response_callback = response_callback
        self._stop_event = threading.Event()
        self.logger = get_logger("STT")
        
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
        """Handle partial STT results with reduced logging"""
        # Only log longer partial results to reduce spam
        if len(text.split()) >= 3:  # At least 3 words
            self.logger.debug(f"Partial: '{text}'")

    def on_final_result(self, text):
        """Handle final STT results and generate chatbot response"""
        if not text.strip():
            return
        
        self.logger.info(f"Final result: '{text}'")
        
        # Check for sleep keywords first
        if self.state_manager and self.state_manager.check_sleep_keywords(text):
            self.logger.info("Sleep keyword detected - going to STANDBY")
            return
        
        # Update activity timestamp
        if self.state_manager:
            self.state_manager.update_activity()
        
        # Generate chatbot response
        try:
            self.logger.debug("Generating chatbot response...")
            response = self.chatbot.get_response(text)
            self.logger.info(f"Bot response: {response}")
            
            # Send response to callback (e.g., for TTS)
            if self.response_callback:
                self.response_callback(response)
                
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            if self.state_manager:
                self.state_manager.transition_to(SystemState.LISTENING, "STT error occurred")

    def run(self):
        self.logger.info("Starting STT + Chatbot thread...")
        
        if self.audio_queue:
            # Process audio from queue (when integrated with VAD)
            while not self._stop_event.is_set():
                try:
                    # Only process if system is active
                    if self.state_manager and not self.state_manager.is_active():
                        time.sleep(0.1)
                        continue
                    
                    # Get audio frame from queue with timeout
                    audio_frame = self.audio_queue.get(timeout=0.5)
                    if audio_frame is not None:
                        self.process_audio_frame(audio_frame)
                except:
                    # Check for timeout and handle conversation end
                    if self.state_manager:
                        self.state_manager.check_timeout()
                    continue
        else:
            # Process audio directly from microphone
            mic_stream = MicStream(samplerate=16000, channels=1, frame_duration_ms=250)
            
            try:
                for frame in mic_stream.stream():
                    if self._stop_event.is_set():
                        break
                    self.process_audio_frame(frame)
            except Exception as e:
                self.logger.error(f"Error in audio processing: {e}")
        
        self.logger.info("STT + Chatbot thread stopped.")

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
