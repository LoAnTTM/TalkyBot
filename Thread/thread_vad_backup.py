import threading
import time
import numpy as np

from components.vad import VoiceActivityDetector
from components.logger import get_logger
from components.state_manager import SystemState


class VADThread(threading.Thread):
    def __init__(self, vad, audio_stream, state_manager=None, audio_queue=None, 
                 status_callback=None, tts_interrupt_event=None, 
                 tts_playing_event=None, tts_audio_ref_callback=None):
        super().__init__()
        self.vad = vad
        self.audio_stream = audio_stream
        self.state_manager = state_manager
        self.audio_queue = audio_queue  # Queue để gửi audio frames cho STT
        self.status_callback = status_callback
        
        # TTS interrupt coordination
        self.tts_interrupt_event = tts_interrupt_event
        self.tts_playing_event = tts_playing_event
        self.tts_audio_ref_callback = tts_audio_ref_callback
        
        self._stop_event = threading.Event()
        self.last_speaking_state = None
        self.frame_count = 0
        self.logger = get_logger("VAD")
        self._vad_failure_count = 0
        self._max_vad_failures = 10  # Max consecutive VAD failures before giving up
        self._fallback_mode = False
        
        # Enhanced VAD state
        self._speaking_start_time = None
        self._last_speech_time = None
        self._audio_buffer = []
        
        # TTS interrupt detection
        self._baseline_noise_level = 50.0  # Will be calibrated
        self._interrupt_threshold_multiplier = 4.0
        self._echo_suppression_enabled = True
        self._tts_suppress_time = 0

    def run(self):
        self.logger.info("VAD Thread started with TTS interrupt capability")
        
        # Quick calibration without hanging
        self.logger.info("🔧 Quick baseline calibration...")
        try:
            # Set reasonable default immediately
            self._baseline_noise_level = 64.0  # Based on your previous output
            self.logger.info(f"📊 Using baseline noise level: {self._baseline_noise_level}")
        except Exception as e:
            self.logger.warning(f"⚠️ Calibration error: {e}")
            self._baseline_noise_level = 50.0
        
        # Continue with existing VAD logic but add TTS awareness
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
                        current_time = time.time()
                        
                        # TTS suppression logic
                        tts_is_speaking = (self.tts_playing_event and self.tts_playing_event.is_set())
                        if tts_is_speaking:
                            self._tts_suppress_time = current_time
                            continue
                        if (self._tts_suppress_time > 0 and current_time - self._tts_suppress_time < 0.5):
                            continue
                        
                        # Process audio frame with echo cancellation
                        processed_frame = self._apply_echo_cancellation(frame)
                        
                        # Log audio stream health every 160 frames (~5 seconds)
                        if test_frame_count % 160 == 0:
                            raw_level = abs(frame).mean() if frame is not None and hasattr(frame, 'mean') else 0.0
                            proc_level = abs(processed_frame).mean() if processed_frame is not None and hasattr(processed_frame, 'mean') else 0.0
                            self.logger.debug(f"VAD: raw={raw_level:.1f}, processed={proc_level:.1f}, tts_playing={tts_is_speaking}")
                        
                        # Process VAD with enhanced interrupt detection
                        try:
                            self.vad.process_frame(processed_frame)
                            info = self.vad.get_continuous_speech_info()
                            current_speaking = info['is_speaking']
                            
                            # Add TTS interrupt detection
                            if current_speaking and tts_is_speaking:
                                audio_level = abs(processed_frame).mean() if processed_frame is not None and hasattr(processed_frame, 'mean') else 0.0
                                interrupt_threshold = self._baseline_noise_level * self._interrupt_threshold_multiplier
                                if audio_level > interrupt_threshold:
                                    self.logger.info(f"🛑 TTS interrupt detected (level: {audio_level:.1f} > {interrupt_threshold:.1f})")
                                    if self.tts_interrupt_event:
                                        self.tts_interrupt_event.set()
                            
                            # Continue with existing VAD logic
                            self._handle_speech_state_change(current_speaking, info)
                            
                            # Send audio frame to STT when active and speaking
                            if self._should_send_audio(current_speaking):
                                self.audio_queue.put(processed_frame)  # Use processed frame
                                if self.frame_count % 50 == 0:
                                    self.logger.debug(f"Sent audio frame to STT queue (queue size: {self.audio_queue.qsize()})")
                            
                            # Call status callback with enhanced info
                            if (current_speaking != self.last_speaking_state or 
                                (current_speaking and self.frame_count % 10 == 0)):
                                if self.status_callback:
                                    # Add interrupt info to callback
                                    enhanced_info = info.copy()
                                    if current_speaking:
                                        audio_level = abs(processed_frame).mean() if processed_frame is not None and hasattr(processed_frame, 'mean') else 0.0
                                        enhanced_info['audio_level'] = audio_level
                                        enhanced_info['interrupt_threshold'] = self._baseline_noise_level * self._interrupt_threshold_multiplier
                                    self.status_callback(enhanced_info)
                                self.last_speaking_state = current_speaking
                        
                        except Exception as vad_error:
                            self._vad_failure_count += 1
                            if self._vad_failure_count <= self._max_vad_failures:
                                self.logger.warning(f"VAD processing error ({self._vad_failure_count}/{self._max_vad_failures}): {vad_error}")
                            
                            if self._vad_failure_count >= self._max_vad_failures and not self._fallback_mode:
                                self.logger.error("Too many VAD failures - entering fallback mode")
                                self._fallback_mode = True
                            
                            # Reset failure count occasionally
                            if self.frame_count % 1000 == 0:
                                self._vad_failure_count = max(0, self._vad_failure_count - 1)
                            
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

    def _calibrate_baseline_noise(self):
        """Calibrate baseline noise level for interrupt detection"""
        self.logger.info("🔧 Calibrating baseline noise level...")
        noise_samples = []
        
        try:
            frame_count = 0
            start_time = time.time()
            
            # Use the same audio stream that will be used in main loop
            for audio_frame in self.audio_stream.stream():
                if hasattr(audio_frame, 'mean'):
                    audio_level = abs(audio_frame).mean()
                elif hasattr(audio_frame, '__len__') and len(audio_frame) > 0:
                    # Calculate mean manually
                    audio_level = sum(abs(x) for x in audio_frame) / len(audio_frame)
                else:
                    audio_level = 0.0
                
                noise_samples.append(audio_level)
                frame_count += 1
                
                # Break after enough samples OR timeout
                if frame_count >= 50 or time.time() - start_time > 3.0:
                    break
            
            if noise_samples:
                mean_noise = sum(noise_samples) / len(noise_samples)
                # Simple std calculation
                variance = sum((x - mean_noise) ** 2 for x in noise_samples) / len(noise_samples)
                std_noise = variance ** 0.5
                self._baseline_noise_level = mean_noise + (2 * std_noise)
                self.logger.info(f"📊 Baseline noise level: {self._baseline_noise_level:.1f} (from {len(noise_samples)} samples)")
            else:
                self._baseline_noise_level = 50.0
                self.logger.warning("⚠️ Could not calibrate baseline - using default: 50.0")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Noise calibration failed: {e}")
            self._baseline_noise_level = 50.0
            
        # Don't continue looping here - return to main processing
        return

    def _apply_echo_cancellation(self, audio_frame):
        """Apply simple echo cancellation using TTS reference audio"""
        if not self._echo_suppression_enabled:
            return audio_frame
        
        # Get TTS reference audio if available
        if self.tts_audio_ref_callback:
            tts_reference = self.tts_audio_ref_callback()
            if tts_reference is not None:
                # Simple spectral subtraction for echo cancellation
                return self._spectral_subtraction(audio_frame, tts_reference)
        
        return audio_frame

    def _spectral_subtraction(self, input_audio, reference_audio):
        """Simple spectral subtraction for echo cancellation"""
        try:
            # Convert to numpy-like operations without importing numpy
            input_list = input_audio.tolist() if hasattr(input_audio, 'tolist') else list(input_audio)
            ref_list = reference_audio.tolist() if hasattr(reference_audio, 'tolist') else list(reference_audio)
            
            # Ensure same length
            min_len = min(len(input_list), len(ref_list))
            
            # Simple amplitude-based suppression
            suppression_factor = 0.3  # Suppress 30% of reference signal
            result = []
            for i in range(min_len):
                suppressed = input_list[i] - (suppression_factor * ref_list[i])
                # Prevent over-suppression
                result.append(max(-32768, min(32767, suppressed)))
            
            # Convert back to original format
            if hasattr(input_audio, '__class__'):
                try:
                    return input_audio.__class__(result)
                except:
                    return result
            return result
            
        except Exception as e:
            self.logger.debug(f"Echo cancellation failed: {e}")
            return input_audio

    def _handle_speech_detected(self, audio_frame, current_time, audio_level, 
                              is_user_interrupt, interrupt_threshold, info):
        """Handle when speech is detected with interrupt capability"""
        
        # State transition for new speech
        if self._speaking_start_time is None:
            self._speaking_start_time = current_time
            self.logger.debug("🎤 Speech started")
            
            # State transition for new speech
            if self.state_manager:
                current_state = self.state_manager.get_current_state()
                if current_state in [SystemState.STANDBY, SystemState.LISTENING]:
                    self.state_manager.transition_to(SystemState.LISTENING, "VAD detected speech")
        
        # Handle TTS interrupt
        if is_user_interrupt and self.tts_interrupt_event:
            self.logger.info(f"🛑 TTS interrupt triggered (level: {audio_level:.1f} > {interrupt_threshold:.1f})")
            self.tts_interrupt_event.set()
            # Continue processing as normal speech
        
        # Buffer audio for STT
        self._audio_buffer.append(audio_frame)
        self._last_speech_time = current_time
        
        # Send audio to STT if in appropriate state
        if self.state_manager:
            current_state = self.state_manager.get_current_state()
            if current_state in [SystemState.LISTENING, SystemState.PROCESSING]:
                try:
                    if self.audio_queue:
                        self.audio_queue.put_nowait(audio_frame)
                except:
                    pass  # Queue full, skip frame
        
        # Status callback with interrupt info
        if self.status_callback:
            duration = current_time - self._speaking_start_time
            enhanced_info = info.copy()
            enhanced_info.update({
                'duration': duration,
                'audio_level': audio_level,
                'interrupt_threshold': interrupt_threshold,
                'is_interrupt': is_user_interrupt
            })
            self.status_callback(enhanced_info)

    def _handle_speech_ended(self, current_time, info):
        """Handle when speech ends"""
        if (self._speaking_start_time is not None and 
            self._last_speech_time is not None and 
            current_time - self._last_speech_time > 0.5):  # 500ms silence threshold
            
            duration = current_time - self._speaking_start_time
            self.logger.debug(f"🔇 Speech ended (duration: {duration:.1f}s)")
            
            # Send final audio buffer to STT
            if self._audio_buffer and len(self._audio_buffer) > 10:  # Minimum frames
                self._send_audio_buffer()
            
            # Reset speech tracking
            self._speaking_start_time = None
            self._last_speech_time = None
            self._audio_buffer = []
            
            # Status callback
            if self.status_callback:
                enhanced_info = info.copy()
                enhanced_info.update({
                    'duration': duration
                })
                self.status_callback(enhanced_info)

    def _send_audio_buffer(self):
        """Send accumulated audio buffer to STT"""
        if not self._audio_buffer or not self.audio_queue:
            return
            
        try:
            # Send all buffered frames
            for frame in self._audio_buffer:
                self.audio_queue.put_nowait(frame)
            self.logger.debug(f"📤 Sent {len(self._audio_buffer)} audio frames to STT")
        except Exception as e:
            self.logger.warning(f"Failed to send audio buffer: {e}")

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