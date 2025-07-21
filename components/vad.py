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
                 state_manager=None,
                 sampling_rate: int = 16000,
                 threshold: float = 0.3,
                 min_speech_duration_ms: int = 100,
                 min_silence_duration_ms: int = 300,
                 buffer_duration_ms: int = 1500):
        """
        Initialize VoiceActivityDetector.

        Args:
            state_manager: Optional state manager for system integration.
            sampling_rate (int): Audio sampling rate.
            threshold (float): Silero VAD speech detection threshold (0.0 - 1.0).
            min_speech_duration_ms (int): Minimum speech duration to be considered a segment.
            min_silence_duration_ms (int): Silence duration after speech to end a speaking session.
            buffer_duration_ms (int): Duration of internal audio buffer (in ms).
        """
        self.state_manager = state_manager
        self.sampling_rate = sampling_rate
        self.min_silence_duration_s = min_silence_duration_ms / 1000.0

        print("Loading Silero VAD model from torch hub...")
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True  # Add trust_repo to avoid warnings
                )
                # Unpack necessary utilities
                (self.get_speech_timestamps, _, _, _, _) = utils
                self.vad_parameters = {
                    'threshold': threshold,
                    'min_speech_duration_ms': min_speech_duration_ms
                }
                print("✅ VAD model loaded successfully!")
                
                # Test model with dummy data
                dummy_audio = torch.randn(16000)  # 1 second of dummy audio
                test_result = self.get_speech_timestamps(
                    dummy_audio, 
                    self.model, 
                    sampling_rate=self.sampling_rate,
                    **self.vad_parameters
                )
                print(f"✅ VAD model test successful - detected {len(test_result)} segments in dummy audio")
                break  # Success, exit retry loop
                
            except Exception as e:
                retry_count += 1
                print(f"❌ Error loading VAD model (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    print(f"⏳ Retrying VAD model load in 3 seconds...")
                    time.sleep(3.0)
                else:
                    print("❌ Failed to load VAD model after all retries")
                    raise

        # Use deque for more efficient buffering
        buffer_size = int(self.sampling_rate * buffer_duration_ms / 1000)
        self.speech_buffer = deque(maxlen=buffer_size)

        # State tracking for continuous speaking sessions
        self.is_speaking = False
        self.last_speech_time = 0.0
        self.speech_start_time = 0.0
        
        # Debug counters
        self._debug_counter = 0
        self._process_frame_count = 0

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
        try:
            self._process_frame_count += 1
            
            # Validate input
            if audio_frame is None:
                print(f"⚠️ VAD received None audio frame (count: {self._process_frame_count})")
                return
            
            tensor_frame = self._normalize_audio_frame(audio_frame)
            self.speech_buffer.extend(tensor_frame.tolist())

            # Only process when buffer has enough data
            if len(self.speech_buffer) < 512:
                if self._process_frame_count % 100 == 0:  # Log every ~10 seconds when starting
                    print(f"🔄 VAD buffer building: {len(self.speech_buffer)}/512 frames")
                return

            buffer_tensor = torch.tensor(list(self.speech_buffer), dtype=torch.float32)
            
            # Calculate audio level for debugging
            audio_level = torch.sqrt(torch.mean(buffer_tensor ** 2)).item()
            
            # Get speech timestamps from VAD model
            try:
                speech_timestamps = self.get_speech_timestamps(
                    buffer_tensor,
                    self.model,
                    sampling_rate=self.sampling_rate,
                    **self.vad_parameters
                )
            except Exception as vad_error:
                print(f"❌ VAD model inference error: {vad_error}")
                return

            current_time = time.time()
            has_speech = len(speech_timestamps) > 0

            # Debug logging - import logger here to avoid circular imports
            try:
                from .logger import get_logger
                logger = get_logger("VAD")
            except:
                logger = None
                
            self._debug_counter += 1
                
            # More frequent debug logging to track VAD health
            if self._debug_counter % 25 == 0:  # Every ~2.5 seconds
                debug_msg = f"VAD Debug - Audio level: {audio_level:.4f}, Speech detected: {has_speech}, Threshold: {self.vad_parameters['threshold']}"
                if logger:
                    logger.info(debug_msg)
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
                    
        except Exception as e:
            print(f"❌ Critical error in VAD process_frame: {e}")
            # Don't crash the thread, just log and continue

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