import threading
import time
import queue

from components.vad import VoiceActivityDetector
from components.logger import get_logger
from audio.mic_stream import AudioStream


class VADThread(threading.Thread):
    def __init__(self, state_manager=None, audio_queue=None, tts_interrupt_event=None, 
                 tts_playing_event=None, tts_audio_ref_callback=None):
        super().__init__()
        self.state_manager = state_manager
        self.audio_queue = audio_queue or queue.Queue()
        self.vad = VoiceActivityDetector(state_manager=state_manager)
        self.audio_stream = AudioStream()
        
        # TTS interrupt coordination
        self.tts_interrupt_event = tts_interrupt_event
        self.tts_playing_event = tts_playing_event
        self.reference_audio = None
        
        self._stop_event = threading.Event()
        self.last_speaking_state = None
        self.frame_count = 0
        self.logger = get_logger("VAD")
        
        # Simple baseline noise level
        self._baseline_noise_level = 50.0
        
    def run(self):
        """Main VAD processing loop - Audio gatekeeper"""
        self.logger.info("VAD Thread started as audio gatekeeper")
        
        # Quick baseline calibration
        self._calibrate_baseline()
        
        try:
            while not self._stop_event.is_set():
                try:
                    for frame in self.audio_stream.stream():
                        if self._stop_event.is_set():
                            break
                        
                        self.frame_count += 1
                        
                        # Process VAD on ALL audio (gatekeeper function)
                        try:
                            self.vad.process_frame(frame)
                            info = self.vad.get_continuous_speech_info()
                            current_speaking = info['is_speaking']
                            
                            # Debug logging every 100 frames to see if VAD is working
                            if self.frame_count % 100 == 0:
                                audio_level = abs(frame).mean() if frame is not None and hasattr(frame, 'mean') else 0.0
                                self.logger.debug(f"📊 VAD Gatekeeper Frame {self.frame_count}: level={audio_level:.1f}, speaking={current_speaking}")
                            
                            # Only forward audio to STT if speech is detected AND system is listening
                            if current_speaking:
                                # Check if system is in listening mode
                                if (self.state_manager and 
                                    hasattr(self.state_manager, 'conversation_active') and 
                                    self.state_manager.conversation_active.is_set()):
                                    
                                    # Skip if TTS is playing (avoid loopback)
                                    if not (self.tts_playing_event and self.tts_playing_event.is_set()):
                                        # Forward to STT
                                        if self.audio_queue:
                                            self.audio_queue.put(frame)
                            
                            # Log state changes
                            if current_speaking != self.last_speaking_state:
                                self.last_speaking_state = current_speaking
                                if current_speaking:
                                    self.logger.info("🗣️ VAD: Speech detected")
                                    if (self.state_manager and 
                                        hasattr(self.state_manager, 'conversation_active') and 
                                        self.state_manager.conversation_active.is_set()):
                                        self.logger.info("📤 Forwarding audio to STT queue")
                                else:
                                    self.logger.info("🔇 VAD: Speech ended")
                                
                        except Exception as e:
                            self.logger.debug(f"VAD frame processing error: {e}")
                            
                except Exception as e:
                    self.logger.error(f"VAD stream error: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            self.logger.error(f"Critical VAD error: {e}")
        finally:
            self.logger.info("VAD Thread stopped")

    def _calibrate_baseline(self):
        """Quick baseline noise calibration"""
        try:
            self.logger.info("🔧 Quick baseline calibration...")
            
            # Simple calibration - just use a default level
            self._baseline_noise_level = 64.0
            
            self.logger.info(f"📊 Using baseline noise level: {self._baseline_noise_level}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Calibration error: {e}")
            self._baseline_noise_level = 50.0

    def set_reference_audio(self, audio_data):
        """Set reference audio for echo cancellation"""
        self.reference_audio = audio_data

    def stop(self):
        """Stop VAD thread"""
        self._stop_event.set()


if __name__ == "__main__":
    # Test VAD thread independently
    from components.state_manager import StateManager
    
    state_manager = StateManager()
    audio_queue = queue.Queue()
    
    vad_thread = VADThread(
        state_manager=state_manager,
        audio_queue=audio_queue
    )
    
    print("🎤 VAD Thread test started!")
    print("=" * 50)
    print("Speak to test voice activity detection...")
    
    vad_thread.start()
    
    try:
        while True:
            try:
                audio_data = audio_queue.get(timeout=1.0)
                print("📊 Audio detected in queue")
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        print("🛑 Stopping VAD thread...")
        vad_thread.stop()
        vad_thread.join()
        print("✅ VAD thread stopped successfully")
