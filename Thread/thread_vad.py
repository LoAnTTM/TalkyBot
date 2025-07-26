import threading
import numpy as np
import sounddevice as sd
from components.vad import VoiceActivityDetector
from components.logger import get_logger
from audio.mic_stream import MicStream
from components.state_manager import SystemState

class VADThread(threading.Thread):
    """
    VADThread: Xử lý audio real-time, phát hiện speech và điều phối pipeline:
    - Wakeword: truyền frame cho wakeword khi ở STANDBY
    - STT: truyền frame trực tiếp cho handler khi ở LISTENING
    - TTS: dừng phát nếu phát hiện speech khi đang phát TTS (dùng event)
    """
    def __init__(self, state_manager=None, wakeword_detector=None, 
                 stt_handler=None, tts_interrupt_event=None, tts_playing_event=None):
        super().__init__()
        self.state_manager = state_manager
        self.wakeword_detector = wakeword_detector
        self.stt_handler = stt_handler
        self.tts_interrupt_event = tts_interrupt_event
        self.tts_playing_event = tts_playing_event
        self.reference_audio = None  
        self.vad = VoiceActivityDetector()
        self._stop_event = threading.Event()
        self.last_speaking_state = None
        self.frame_count = 0
        self.logger = get_logger("VAD")

    def run(self):
        self.logger.info("VAD Thread started as audio gatekeeper")
        try:
            mic_stream = MicStream(samplerate=16000,
                                    channels=1,
                                    frame_duration_ms=500)
            for frame in mic_stream.stream():
                if self._stop_event.is_set():
                    break
                self.frame_count += 1
                try:
                    self.vad.process_frame(frame)
                    info = self.vad.get_continuous_speech_info()
                    current_speaking = info['is_speaking']

                    # Wakeword pipeline
                    if current_speaking and self._is_state(SystemState.STANDBY) and self.wakeword_detector:
                        frame = np.array(frame, dtype=np.float32).reshape(-1, 1)
                        if frame.max() > 1.0 or frame.min() < -1.0:
                            frame = frame / 32767.0
                        mean_amp = np.abs(frame).mean()
                        self.logger.debug(f"Wakeword frame stats: min={frame.min()}, max={frame.max()}, mean_amp={mean_amp:.6f}, dtype={frame.dtype}, shape={frame.shape}, first5={frame[:5].flatten()}")
                        self.wakeword_detector.process_frame(frame)

                    # STT pipeline (gọi trực tiếp, không queue)
                    if current_speaking and self._is_state('LISTENING') and self.stt_handler:
                        
                        self.stt_handler.process_frame(frame)

                    # TTS interrupt pipeline
                    if current_speaking and self.tts_playing_event and self.tts_playing_event.is_set():
                        if self.tts_interrupt_event:
                            self.logger.info("🛑 VAD: Interrupting TTS due to user speech")
                            self.tts_interrupt_event.set()
                        try:
                            sd.stop()
                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to stop sounddevice: {e}")

                    # Log state changes
                    if current_speaking != self.last_speaking_state:
                        self.last_speaking_state = current_speaking
                        if current_speaking:
                            self.logger.info("🗣️ VAD: Speech detected")
                        else:
                            self.logger.info("🔇 VAD: Silence!!!!!")

                    # Debug logging
                    if self.frame_count % 100 == 0:
                        audio_level = abs(frame).mean() if frame is not None and hasattr(frame, 'mean') else 0.0
                        self.logger.debug(f"📊 VAD Gatekeeper Frame {self.frame_count}: level={audio_level:.1f}, speaking={current_speaking}")
                except Exception as e:
                    self.logger.debug(f"VAD frame processing error: {e}")
        except Exception as e:
            self.logger.error(f"Critical VAD error: {e}")
        finally:
            self.logger.info("VAD Thread stopped")

    def _is_state(self, state_name):
        return (
            self.state_manager is not None
            and hasattr(self.state_manager, 'get_current_state')
            and self.state_manager.get_current_state() == state_name
        )

    def set_reference_audio(self, audio_data):
        """Set reference audio for echo cancellation"""
        self.reference_audio = audio_data

    def stop(self):
        """Stop VAD thread"""
        self._stop_event.set()
