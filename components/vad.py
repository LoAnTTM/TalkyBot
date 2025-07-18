from audio.mic_stream import AudioStream

import time
from collections import deque
from typing import List, Dict, Optional

import numpy as np
import torch


class VoiceActivityDetector:
    """
    Class to detect voice activity (VAD) in real-time audio streams
    using the Silero VAD model.
    """
    def __init__(self,
                 sampling_rate: int = 16000,
                 threshold: float = 0.3,
                 min_speech_duration_ms: int = 100,
                 min_silence_duration_ms: int = 300,
                 buffer_duration_ms: int = 1500):
        """
        Initialize VoiceActivityDetector.

        Args:
            sampling_rate (int): Audio sampling rate.
            threshold (float): Silero VAD speech detection threshold (0.0 - 1.0).
            min_speech_duration_ms (int): Minimum speech duration to be considered a segment.
            min_silence_duration_ms (int): Silence duration after speech to end a speaking session.
            buffer_duration_ms (int): Duration of internal audio buffer (in ms).
        """
        self.sampling_rate = sampling_rate
        self.min_silence_duration_s = min_silence_duration_ms / 1000.0

        print("Loading Silero VAD model from torch hub...")
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            # Unpack necessary utilities
            (self.get_speech_timestamps, _, _, _, _) = utils
            self.vad_parameters = {
                'threshold': threshold,
                'min_speech_duration_ms': min_speech_duration_ms
            }
            print("VAD model loaded successfully!")
        except Exception as e:
            print(f"Error loading VAD model: {e}")
            raise

        # Use deque for more efficient buffering
        buffer_size = int(self.sampling_rate * buffer_duration_ms / 1000)
        self.speech_buffer = deque(maxlen=buffer_size)

        # State tracking for continuous speaking sessions
        self.is_speaking = False
        self.last_speech_time = 0.0
        self.speech_start_time = 0.0

    def _normalize_audio_frame(self, audio_frame: np.ndarray) -> torch.Tensor:
        """Normalize input audio frame to torch.Tensor float32 format."""
        if not isinstance(audio_frame, np.ndarray):
            audio_frame = np.array(audio_frame, dtype=np.float32)

        if audio_frame.dtype == np.int16:
            audio_frame = audio_frame.astype(np.float32) / 32768.0
        elif audio_frame.dtype != np.float32:
            audio_frame = audio_frame.astype(np.float32)

        if audio_frame.ndim > 1:
            audio_frame = audio_frame.flatten()
        
        # Convert to PyTorch Tensor
        return torch.from_numpy(audio_frame)

    def process_frame(self, audio_frame: np.ndarray):
        """
        Process an audio frame to update speech detection state.
        This method only updates internal state, does not return a value.
        """
        tensor_frame = self._normalize_audio_frame(audio_frame)
        self.speech_buffer.extend(tensor_frame.tolist())

        # Only process when buffer has enough data
        if len(self.speech_buffer) < 512:
            return

        buffer_tensor = torch.tensor(list(self.speech_buffer), dtype=torch.float32)
        
        # Calculate audio level for debugging
        audio_level = torch.sqrt(torch.mean(buffer_tensor ** 2)).item()
        
        speech_timestamps = self.get_speech_timestamps(
            buffer_tensor,
            self.model,
            sampling_rate=self.sampling_rate,
            **self.vad_parameters
        )

        current_time = time.time()
        has_speech = len(speech_timestamps) > 0

        # Debug logging - import logger here to avoid circular imports
        try:
            from .logger import get_logger
            logger = get_logger("VAD")
        except:
            logger = None
            
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        # Only log when audio level is significant or speech state changes
        if self._debug_counter % 50 == 0 and (audio_level > 0.02 or has_speech):  # Every ~5 seconds, only when relevant
            debug_msg = f"Audio: {audio_level:.3f} {'🎤 Speech' if has_speech else '🔇 Quiet'}"
            if logger:
                logger.debug(debug_msg)  # Changed to debug level
            else:
                print(f"🔍 {debug_msg}")

        if has_speech:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = current_time
                speech_msg = f"Speech started! (audio_level: {audio_level:.4f})"
                if logger:
                    logger.info(speech_msg)
                else:
                    print(f"🎤 {speech_msg}")
            self.last_speech_time = current_time
        elif self.is_speaking and (current_time - self.last_speech_time) > self.min_silence_duration_s:
            self.is_speaking = False
            end_msg = f"Speech ended after {current_time - self.speech_start_time:.1f}s"
            if logger:
                logger.info(end_msg)
            else:
                print(f"🔇 {end_msg}")

    def get_continuous_speech_info(self) -> Dict:
        """
        Return information about the current continuous speaking session.
        """
        if self.is_speaking:
            duration = time.time() - self.speech_start_time
            return {
                'is_speaking': True,
                'duration': duration,
                'start_time': self.speech_start_time
            }
        return {'is_speaking': False, 'duration': 0, 'start_time': None}

    def reset_speech_state(self):
        """Reset speech detection state."""
        self.speech_buffer.clear()
        self.is_speaking = False
        self.last_speech_time = 0.0
        self.speech_start_time = 0.0

def test_continuous_vad():
    vad = VoiceActivityDetector(sampling_rate=16000, threshold=0.5, min_speech_duration_ms=100)
    audio_stream = AudioStream(samplerate=16000, channels=1, frame_duration_ms=100)

    print("🎤 Starting continuous VAD test...")
    print("Speak long sentences to test (Press Ctrl+C to stop)")
    print("=" * 50)

    try:
        frame_count = 0
        for audio_frame in audio_stream.stream():
            frame_count += 1
            
            # 1. Process audio frame to update VAD state
            vad.process_frame(audio_frame)
            
            # 2. Get updated state information
            speech_info = vad.get_continuous_speech_info()
            
            # Calculate audio level for display
            audio_level = np.abs(audio_frame).mean()
            
            # Display information every 5 frames (500ms)
            if frame_count % 5 == 0:
                if speech_info['is_speaking']:
                    duration = speech_info['duration']
                    print(f"🎤 SPEAKING ({duration:.1f}s) | Audio level: {audio_level:.4f}")
                else:
                    print(f"🔇 SILENT      | Audio level: {audio_level:.4f}")
            
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("🛑 Stopping VAD test")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    test_continuous_vad()