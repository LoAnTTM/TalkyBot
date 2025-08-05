import threading
import numpy as np
import sounddevice as sd
from collections import deque

from components.vad import VoiceActivityDetector
from components.logger import get_logger
from audio.mic_stream import MicStream
from components.state_manager import SystemState


class VADThread(threading.Thread):
    """
    VADThread: Real-time audio processing, speech detection and pipeline coordination:
    - Wakeword: send frame to wakeword when in STANDBY
    - STT: send frame directly to handler when in LISTENING
    - TTS: stop playback if speech detected during TTS (using event)
    """
    def __init__(self, mic_stream, state_manager=None, wakeword_thread=None, 
                 stt_thread=None, tts_interrupt_event=None, tts_playing_event=None):
        super().__init__()
        self.mic_stream = mic_stream
        self.state_manager = state_manager
        self.wakeword_thread = wakeword_thread
        self.stt_thread = stt_thread
        self.tts_interrupt_event = tts_interrupt_event
        self.tts_playing_event = tts_playing_event
        self.reference_audio = None  
        self.vad = VoiceActivityDetector()
        self._stop_event = threading.Event()
        self.last_speaking_state = None
        self.frame_count = 0
        self.logger = get_logger("VAD")
        
        # Pre-speech buffer: keep frames before speech is detected
        self.pre_speech_buffer = deque(maxlen=10)  # about 5s if 500ms per frame

    def run(self):
        self.logger.info("VAD Thread started as audio gatekeeper")
        try:
            for frame in self.mic_stream.stream():
                if self._stop_event.is_set():
                    break
                self.frame_count += 1
                try:
                    self.vad.process_frame(frame)
                    info = self.vad.get_continuous_speech_info()
                    current_speaking = info['is_speaking']

                    # Always save to pre-speech buffer
                    self.pre_speech_buffer.append(frame)

                    # If just switched from silence to speech -> flush buffered frames
                    if current_speaking and self.last_speaking_state is False:
                        self.logger.info("🗣️ VAD: Speech started — flushing buffered frames")
                        for buffered_frame in list(self.pre_speech_buffer):
                            if self._is_state(SystemState.STANDBY) and self.wakeword_thread:
                                self.wakeword_thread.process_frame(buffered_frame)
                            elif self._is_state(SystemState.LISTENING) and self.stt_thread:
                                self.stt_thread.process_frame(buffered_frame)
                        self.pre_speech_buffer.clear()

                    # Clear pre-speech buffer when system state changes
                    if not current_speaking and self.last_speaking_state is True:
                        self.logger.info("🔇 VAD: Silence detected — clearing pre-speech buffer")
                        self.pre_speech_buffer.clear()

                    # If speaking,
                    if current_speaking:
                        if self._is_state(SystemState.STANDBY):
                            self.wakeword_thread.process_frame(frame)
                        elif self._is_state(SystemState.LISTENING):
                            self.stt_thread.process_frame(frame)

                        # If TTS is playing, interrupt immediately
                        if self.tts_playing_event and self.tts_playing_event.is_set():
                            if self.tts_interrupt_event:
                                self.logger.info("🛑 VAD: Interrupting TTS due to user speech")
                                self.tts_interrupt_event.set()
                            try:
                                sd.stop()
                            except Exception as e:
                                self.logger.warning(f"⚠️ Failed to stop sounddevice: {e}")

                    # Log state change
                    if current_speaking != self.last_speaking_state:
                        self.last_speaking_state = current_speaking
                        if current_speaking:
                            self.logger.info("🗣️ VAD: Speech detected")
                        else:
                            self.logger.info("🔇 VAD: Silence!!!!!")

                    # Debug logging
                    if self.frame_count % 100 == 0:
                        audio_level = abs(frame).mean() if hasattr(frame, 'mean') else 0.0
                        self.logger.debug(f"📊 VAD Frame {self.frame_count}: level={audio_level:.1f}, speaking={current_speaking}")
                except Exception as e:
                    self.logger.debug(f"VAD frame processing error: {e}")
        except Exception as e:
            self.logger.error(f"Critical VAD error: {e}")
        finally:
            self.logger.info("VAD Thread stopped")

    def _is_state(self, target_state):
        """Check if current system state matches the target"""
        try:
            self.logger.debug(f"_is_state: current={self.state_manager.current_state}, compare={target_state}")
            return self.state_manager and self.state_manager.current_state == target_state
            
        except Exception as e:
            self.logger.warning(f"⚠️ State check failed: {e}")
            return False



    def set_reference_audio(self, audio_data):
        """Set reference audio for echo cancellation"""
        self.reference_audio = audio_data

    def stop(self):
        """Stop VAD thread"""
        self._stop_event.set()
