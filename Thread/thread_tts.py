import threading
import queue
import time
import subprocess
import tempfile
import os

from components.tts import TextToSpeech
from components.logger import get_logger
from components.state_manager import SystemState


class TTSThread(threading.Thread):
    def __init__(self, state_manager=None, max_queue_size=10, interrupt_event=None, 
                 playing_event=None, audio_callback=None):
        super().__init__()
        self.tts = TextToSpeech()
        self.text_queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self.is_speaking = False
        self.state_manager = state_manager
        self.logger = get_logger("TTS")
        
        # TTS interrupt coordination
        self.interrupt_event = interrupt_event
        self.playing_event = playing_event
        self.audio_callback = audio_callback  # Called when audio starts playing
        self.current_process = None

    def run(self):
        self.logger.info("TTS Thread started with interrupt capability")
        
        while not self._stop_event.is_set():
            try:
                # Check if we should process TTS queue based on state
                if self.state_manager and not self.state_manager.conversation_active.is_set():
                    time.sleep(0.1)
                    continue
                
                # Wait for new text to speak, timeout to check stop event
                text = self.text_queue.get(timeout=0.5)
                if text and not self._stop_event.is_set():
                    self._speak_with_interrupt_support(text)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"TTS Error: {e}")
                self.is_speaking = False
                if self.state_manager:
                    self.state_manager.transition_to(SystemState.LISTENING, "TTS error occurred")

        self.logger.info("TTS Thread stopped")
    
    def _speak_with_interrupt_support(self, text):
        """Speak text with interrupt capability"""
        try:
            self.logger.info(f"🗣️ Speaking: {text[:50]}...")
            self.is_speaking = True
            
            # Generate audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                self.tts.tts.tts_to_file(text=text, file_path=tmp_file.name)
                audio_file_path = tmp_file.name
            
            # Load audio data for reference (echo cancellation)
            if self.audio_callback:
                try:
                    # Simple audio data loading without numpy
                    with open(audio_file_path, 'rb') as f:
                        # Skip WAV header (44 bytes) and read raw audio data
                        f.seek(44)
                        audio_data = f.read()
                        # Convert to simple list for reference
                        audio_ref = list(audio_data)
                        self.audio_callback(audio_ref)
                except Exception as e:
                    self.logger.debug(f"Could not load audio reference: {e}")
            
            # Signal that TTS is starting
            if self.playing_event:
                self.playing_event.set()
            
            # Play audio with interrupt monitoring
            self._play_audio_with_interrupt(audio_file_path)
            
        except Exception as e:
            self.logger.error(f"Error in TTS speak: {e}")
        finally:
            # Clear playing state
            if self.playing_event:
                self.playing_event.clear()
            
            self.is_speaking = False
            
            # Clean up temp file
            if 'audio_file_path' in locals():
                try:
                    os.unlink(audio_file_path)
                except:
                    pass

    def _play_audio_with_interrupt(self, audio_file_path):
        """Play audio file with interrupt monitoring"""
        try:
            if self.state_manager:
                self.state_manager.transition_to(SystemState.SPEAKING, "TTS started")
            
            # Start audio playback process (macOS)
            self.current_process = subprocess.Popen([
                'afplay', audio_file_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Monitor for interrupts while playing
            while self.current_process.poll() is None:
                # Check for interrupt signal
                if (self.interrupt_event and self.interrupt_event.is_set()) or self._stop_event.is_set():
                    self.logger.info("🛑 TTS interrupted - stopping playback")
                    self.current_process.terminate()
                    
                    # Clear interrupt flag
                    if self.interrupt_event:
                        self.interrupt_event.clear()
                    break
                    
                time.sleep(0.05)  # Check every 50ms for responsive interruption
            
            # Ensure process cleanup
            if self.current_process.poll() is None:
                self.current_process.kill()
                
        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")
        finally:
            self.current_process = None
            if self.state_manager:
                self.state_manager.transition_to(SystemState.LISTENING, "TTS completed")

    def speak_text(self, text, priority=False):
        """Add text to queue for speech synthesis"""
        if not text or not text.strip():
            return False
        
        # Don't add to queue if system is not active
        if self.state_manager and not self.state_manager.conversation_active.is_set():
            self.logger.debug(f"Ignoring TTS request - system not active: '{text[:30]}...'")
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
                    self.interrupt_current_speech()
            
            self.text_queue.put(text.strip(), timeout=0.1)
            self.logger.debug(f"Added to TTS queue: '{text[:50]}...'")
            return True
        except queue.Full:
            self.logger.warning("TTS queue is full, dropping message")
            return False

    def interrupt_current_speech(self):
        """Interrupt current TTS speech immediately"""
        if self.interrupt_event:
            self.interrupt_event.set()
        if self.current_process:
            self.current_process.terminate()
        self.logger.info("TTS speech interrupted")

    def stop(self):
        """Stop TTS thread"""
        self._stop_event.set()
        if self.current_process:
            self.current_process.terminate()

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
