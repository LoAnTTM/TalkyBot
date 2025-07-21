import threading
import time

from components.vad import VoiceActivityDetector
from components.logger import get_logger
from components.state_manager import SystemState

class VADThread(threading.Thread):
    def __init__(self, vad, audio_stream, state_manager=None, audio_queue=None, status_callback=None):
        super().__init__()
        self.vad = vad
        self.audio_stream = audio_stream
        self.state_manager = state_manager
        self.audio_queue = audio_queue  # Queue để gửi audio frames cho STT
        self.status_callback = status_callback
        self._stop_event = threading.Event()
        self.last_speaking_state = None
        self.frame_count = 0
        self.logger = get_logger("VAD")
        self._vad_failure_count = 0
        self._max_vad_failures = 10  # Max consecutive VAD failures before giving up
        self._fallback_mode = False

    def run(self):
        self.logger.info("VAD Thread started")
        
        try:
            # Test audio stream first
            self.logger.info("Testing audio stream...")
            test_frame_count = 0
            
            while not self._stop_event.is_set():
                try:
                    for frame in self.audio_stream.stream():
                        if self._stop_event.is_set():
                            break
                        
                        self.frame_count += 1
                        test_frame_count += 1
                        
                        # Log audio stream health every 50 frames (~5 seconds)
                        if test_frame_count % 50 == 0:
                            audio_level = abs(frame).mean() if frame is not None and hasattr(frame, 'mean') else 0.0
                            self.logger.info(f"VAD Debug - Audio level: {audio_level:.4f}, Frame count: {test_frame_count}")
                        
                        # Process VAD
                        try:
                            self.vad.process_frame(frame)
                            info = self.vad.get_continuous_speech_info()
                            current_speaking = info['is_speaking']
                            
                            # Handle state transitions and interrupts
                            self._handle_speech_state_change(current_speaking, info)
                            
                            # Send audio frame to STT when active and speaking
                            if self._should_send_audio(current_speaking):
                                self.audio_queue.put(frame)
                                if self.frame_count % 50 == 0:  # Log every ~5 seconds during speech
                                    self.logger.debug(f"Sent audio frame to STT queue (queue size: {self.audio_queue.qsize()})")
                            
                            # Call status callback on state changes or periodic updates
                            if (current_speaking != self.last_speaking_state or 
                                (current_speaking and self.frame_count % 10 == 0)):
                                if self.status_callback:
                                    self.status_callback(info)
                                self.last_speaking_state = current_speaking
                        
                        except Exception as vad_error:
                            self.logger.error(f"VAD processing error: {vad_error}")
                            # Continue processing even if VAD fails
                            continue
                        
                        time.sleep(0.01)
                        
                except Exception as stream_error:
                    self.logger.error(f"Audio stream error: {stream_error}")
                    # Restart audio stream
                    try:
                        self.logger.info("Attempting to restart audio stream...")
                        self.audio_stream.restart_stream()
                        time.sleep(1.0)  # Wait before retry
                    except Exception as restart_error:
                        self.logger.error(f"Failed to restart audio stream: {restart_error}")
                        time.sleep(5.0)  # Wait longer before retry
        
        except Exception as e:
            self.logger.error(f"Critical VAD thread error: {e}")
        finally:
            self.logger.info("VAD Thread stopped")
    
    def _handle_speech_state_change(self, current_speaking, info):
        """Handle speech state changes and manage system state transitions"""
        if not self.state_manager:
            return
        
        current_state = self.state_manager.get_current_state()
        
        # Handle speech start
        if current_speaking and not self.last_speaking_state:
            self.logger.debug(f"Speech started - Duration: {info['duration']:.1f}s")
            
            # Update activity timestamp
            self.state_manager.update_activity()
            
            # Interrupt TTS if currently speaking
            if current_state == SystemState.SPEAKING:
                self.logger.info("Interrupting TTS - User started speaking")
                self.state_manager.interrupt_speaking("User speech detected")
            
            # Transition to processing if listening
            elif current_state == SystemState.LISTENING:
                self.state_manager.start_processing("Speech detected")
        
        # Handle speech end
        elif not current_speaking and self.last_speaking_state:
            self.logger.debug("Speech ended")
            
            # Update activity timestamp
            self.state_manager.update_activity()
            
            # Return to listening if we were processing
            if current_state == SystemState.PROCESSING:
                self.state_manager.transition_to(SystemState.LISTENING, "Speech ended")
    
    def _should_send_audio(self, is_speaking):
        """Determine if audio should be sent to STT queue"""
        if not is_speaking or not self.audio_queue:
            return False
        
        # Only send audio if system is active
        if self.state_manager:
            return self.state_manager.is_active()
        
        return True

    def stop(self):
        self._stop_event.set()


def vad_status_callback(info):
    """Callback function to handle VAD status updates with reduced spam"""
    if info['is_speaking']:
        print(f"🎤 Speech detected - Duration: {info['duration']:.1f}s")
    else:
        print("🔇 Speech ended - Silence detected")

if __name__ == "__main__":
    from audio.mic_stream import AudioStream
    
    # Create VAD detector and audio stream
    vad = VoiceActivityDetector(sampling_rate=16000, threshold=0.5, min_speech_duration_ms=100)
    audio_stream = AudioStream(samplerate=16000, channels=1, frame_duration_ms=100)
    
    # Create VAD thread with callback to receive status
    vad_thread = VADThread(vad, audio_stream, status_callback=vad_status_callback)
    
    # Start the thread
    vad_thread.start()
    
    print("🎤 VAD Thread test started!")
    print("Speak to test speech detection (Press Ctrl+C to stop)")
    print("=" * 50)

    try:
        # Main thread can do other work or just wait
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("🛑 Stopping program...")
        vad_thread.stop()
        vad_thread.join()
        print("✅ Program stopped.")