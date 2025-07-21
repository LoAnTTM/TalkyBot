import threading
import queue
import time
import signal
import sys

from thread.thread_wakeup import WakeWordThread
from thread.thread_stt import STTConversationThread
from thread.thread_tts import TTSThread
from thread.thread_vad import VADThread
from components.vad import VoiceActivityDetector
from components.state_manager import StateManager, SystemState
from components.logger import setup_logging, get_logger, close_logging
from audio.mic_stream import AudioStream


class TalkyBotSystem:
    """
    Main TalkyBot system with state-driven architecture.
    Manages 4 concurrent threads with centralized state management.
    """
    
    def __init__(self):
        # Initialize logging first
        self.logger_system = setup_logging()
        self.logger = get_logger("MAIN")
        
        # Test audio system before proceeding
        self.logger.info("Testing audio system...")
        try:
            # Test audio stream
            test_audio = AudioStream()
            if test_audio.test_device():
                self.logger.info("✅ Audio system test passed")
            else:
                self.logger.warning("⚠️ Audio system test failed - continuing anyway")
                self.logger.warning("⚠️ You may experience audio issues")
        except Exception as e:
            self.logger.warning(f"⚠️ Audio system test error: {e}")
            self.logger.warning("⚠️ Continuing without audio test - you may experience issues")
        
        # Create state manager
        self.state_manager = StateManager(timeout_seconds=15)
        
        # Create queues for inter-thread communication
        self.audio_to_stt_queue = queue.Queue()
        self.text_to_tts_queue = queue.Queue()
        
        # Initialize audio components
        self.vad = VoiceActivityDetector()
        self.audio_stream = AudioStream()
        
        # Initialize threads
        self._init_threads()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Register state callbacks
        self.state_manager.add_state_callback(self._on_state_change)
        
        self.logger.info("TalkyBot System initialized successfully")
    
    def _init_threads(self):
        """Initialize all system threads"""
        
        # Wake Word Thread - detects "Alexa" to wake system
        self.wakeup_thread = WakeWordThread(state_manager=self.state_manager)
        
        # TTS Thread - handles text-to-speech with interrupt capability
        self.tts_thread = TTSThread(
            state_manager=self.state_manager,
            max_queue_size=10
        )
        self.tts_thread.text_queue = self.text_to_tts_queue
        
        # STT + Brain Thread - handles speech recognition and chatbot
        self.stt_thread = STTConversationThread(
            state_manager=self.state_manager,
            audio_queue=self.audio_to_stt_queue,
            response_callback=self._on_chatbot_response
        )
        
        # VAD Thread - voice activity detection and audio gatekeeper
        self.vad_thread = VADThread(
            vad=self.vad,
            audio_stream=self.audio_stream,
            state_manager=self.state_manager,
            audio_queue=self.audio_to_stt_queue,
            status_callback=self._on_vad_status
        )
    
    def _on_chatbot_response(self, response):
        """Handle chatbot response and send to TTS"""
        self.logger.info(f"Chatbot response: {response[:50]}...")
        if response and response.strip():
            self.tts_thread.speak_text(response)
    
    def _on_vad_status(self, info):
        """Handle VAD status updates with reduced logging"""
        if info['is_speaking']:
            # Only log every 2 seconds during speech to reduce spam
            duration = info['duration']
            if duration > 1.0 and int(duration) % 2 == 0:
                self.logger.debug(f"🎤 Speaking: {duration:.1f}s")
        else:
            # Only log speech end if it was significant (>0.5s)
            if info.get('duration', 0) > 0.5:
                self.logger.debug("🔇 Speech ended")
    
    def _on_state_change(self, old_state, new_state, reason):
        """Handle system state changes"""
        self.logger.info(f"State: {old_state.value} → {new_state.value} ({reason})")
        
        # Log state changes to structured logging
        self.logger_system.log_state_change(old_state.value, new_state.value, reason)
    
    def start(self):
        """Start all system threads"""
        self.logger.info("=" * 60)
        self.logger.info("Starting TalkyBot System...")
        self.logger.info("=" * 60)
        
        try:
            # Start all threads
            self.logger.info("Starting Wake Word thread...")
            self.wakeup_thread.start()
            
            self.logger.info("Starting TTS thread...")
            self.tts_thread.start()
            
            self.logger.info("Starting STT + Brain thread...")
            self.stt_thread.start()
            
            self.logger.info("Starting VAD thread...")
            self.vad_thread.start()
            
            self.logger.info("All threads started successfully!")
            self.logger.info("System ready - Say 'Alexa' to wake up")
            self.logger.info("Sleep commands: bye bye, goodbye, go to sleep, etc.")
            
            # Main system loop
            self._run_system_loop()
            
        except Exception as e:
            self.logger.error(f"Error starting system: {e}")
            self.stop()
    
    def _run_system_loop(self):
        """Main system monitoring loop"""
        try:
            while True:
                # Check for timeout every second
                if self.state_manager.check_timeout():
                    self.logger.info("Conversation timed out - system returned to STANDBY")
                
                # System health monitoring
                self._monitor_system_health()
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.error(f"Error in system loop: {e}")
    
    def _monitor_system_health(self):
        """Monitor system health and log status"""
        # Log periodic status (every 30 seconds when active)
        if hasattr(self, '_last_status_log'):
            if time.time() - self._last_status_log > 30:
                if self.state_manager.is_active():
                    state_info = self.state_manager.get_state_info()
                    self.logger.debug(f"System status: {state_info['current_state']} "
                                    f"(active for {state_info['time_since_activity']:.1f}s)")
                self._last_status_log = time.time()
        else:
            self._last_status_log = time.time()
            
        # Check VAD thread health
        if hasattr(self, 'vad_thread') and self.vad_thread.is_alive():
            # Check if VAD is processing frames
            if hasattr(self.vad_thread, 'frame_count'):
                if not hasattr(self, '_last_vad_frame_count'):
                    self._last_vad_frame_count = self.vad_thread.frame_count
                    self._last_vad_check_time = time.time()
                else:
                    current_time = time.time()
                    if current_time - self._last_vad_check_time > 60:  # Check every minute
                        if self.vad_thread.frame_count == self._last_vad_frame_count:
                            self.logger.error("⚠️ VAD thread appears to be stuck - no new frames processed")
                            # Optionally restart VAD thread here
                            self._restart_vad_thread()
                        else:
                            frames_processed = self.vad_thread.frame_count - self._last_vad_frame_count
                            self.logger.debug(f"✅ VAD thread healthy - processed {frames_processed} frames in last minute")
                        
                        self._last_vad_frame_count = self.vad_thread.frame_count
                        self._last_vad_check_time = current_time
    
    def _restart_vad_thread(self):
        """Restart VAD thread if it gets stuck"""
        try:
            self.logger.info("🔄 Restarting VAD thread...")
            
            # Stop current VAD thread
            if hasattr(self, 'vad_thread') and self.vad_thread.is_alive():
                self.vad_thread.stop()
                self.vad_thread.join(timeout=3)
            
            # Create new VAD thread
            self.vad_thread = VADThread(
                vad=self.vad,
                audio_stream=self.audio_stream,
                state_manager=self.state_manager,
                audio_queue=self.audio_to_stt_queue,
                status_callback=self._on_vad_status
            )
            
            # Start new thread
            self.vad_thread.start()
            self.logger.info("✅ VAD thread restarted successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to restart VAD thread: {e}")

    def stop(self):
        """Stop all system threads gracefully"""
        self.logger.info("=" * 60)
        self.logger.info("Stopping TalkyBot System...")
        self.logger.info("=" * 60)
        
        try:
            # Stop all threads
            self.logger.info("Stopping VAD thread...")
            self.vad_thread.stop()
            
            self.logger.info("Stopping STT thread...")
            self.stt_thread.stop()
            
            self.logger.info("Stopping TTS thread...")
            self.tts_thread.stop()
            
            self.logger.info("Stopping Wake Word thread...")
            self.wakeup_thread.stop()
            
            # Wait for threads to finish
            self.logger.info("Waiting for threads to finish...")
            self.vad_thread.join(timeout=2)
            self.stt_thread.join(timeout=2)
            self.tts_thread.join(timeout=2)
            # WakeWordThread is daemon, will stop automatically
            
            self.logger.info("All threads stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            # Close logging system
            self.logger.info("TalkyBot System stopped cleanly")
            close_logging()
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        self.logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)
    
    def get_status(self):
        """Get current system status"""
        return {
            'state_info': self.state_manager.get_state_info(),
            'thread_status': {
                'vad_alive': self.vad_thread.is_alive() if hasattr(self, 'vad_thread') else False,
                'stt_alive': self.stt_thread.is_alive() if hasattr(self, 'stt_thread') else False,
                'tts_alive': self.tts_thread.is_alive() if hasattr(self, 'tts_thread') else False,
                'wake_alive': self.wakeup_thread.is_alive() if hasattr(self, 'wakeup_thread') else False,
            },
            'queue_sizes': {
                'audio_queue': self.audio_to_stt_queue.qsize(),
                'tts_queue': self.text_to_tts_queue.qsize(),
            }
        }


def main():
    """Main entry point for TalkyBot"""
    try:
        # Create and start TalkyBot system
        system = TalkyBotSystem()
        system.start()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        if 'system' in locals():
            system.stop()


if __name__ == "__main__":
    main()
