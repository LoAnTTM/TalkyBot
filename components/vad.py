import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from audio.mic_stream import MicStream

import time
from collections import deque
from typing import Dict

import numpy as np
import torch

class VoiceActivityDetector:
    def __init__(self,
                 sampling_rate: int = 16000,
                 threshold: float = 0.3,
                 min_speech_duration_ms: int = 100,
                 min_silence_duration_ms: int = 300,
                 buffer_duration_ms: int = 1500):
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
                    trust_repo=True
                )
                (self.get_speech_timestamps, *_ ) = utils
                self.vad_parameters = {
                    'threshold': threshold,
                    'min_speech_duration_ms': min_speech_duration_ms
                }
                print("✅ VAD model loaded successfully!")
                break
            except Exception as e:
                retry_count += 1
                print(f"❌ Error loading VAD model (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    print("⏳ Retrying in 3 seconds...")
                    time.sleep(3.0)
                else:
                    raise
        buffer_size = int(self.sampling_rate * buffer_duration_ms / 1000)
        self.speech_buffer = deque(maxlen=buffer_size)

        self.is_speaking = False
        self.last_speech_time = 0.0
        self.speech_start_time = 0.0

        self._process_frame_count = 0

    def _normalize_audio_frame(self, audio_frame: np.ndarray) -> torch.Tensor:
        if not isinstance(audio_frame, np.ndarray):
            audio_frame = np.array(audio_frame, dtype=np.float32)
        if audio_frame.dtype == np.int16:
            audio_frame = audio_frame.astype(np.float32) / 32768.0
        elif audio_frame.dtype != np.float32:
            audio_frame = audio_frame.astype(np.float32)
        if audio_frame.ndim > 1:
            audio_frame = audio_frame.flatten()
        return torch.from_numpy(audio_frame)

    def process_frame(self, audio_frame: np.ndarray):
        try:
            self._process_frame_count += 1
            if audio_frame is None:
                print(f"⚠️ Received None audio frame ({self._process_frame_count})")
                return
            tensor_frame = self._normalize_audio_frame(audio_frame)
            self.speech_buffer.extend(tensor_frame.tolist())
            if len(self.speech_buffer) < 512:
                if self._process_frame_count % 100 == 0:
                    print(f"🔄 Buffer filling: {len(self.speech_buffer)}/512 samples")
                return
            buffer_tensor = torch.tensor(list(self.speech_buffer), dtype=torch.float32)
            try:
                speech_timestamps = self.get_speech_timestamps(
                    buffer_tensor,
                    self.model,
                    sampling_rate=self.sampling_rate,
                    **self.vad_parameters
                )
            except Exception as vad_error:
                print(f"❌ VAD inference error: {vad_error}")
                return
            current_time = time.time()
            has_speech = len(speech_timestamps) > 0
            if has_speech:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_start_time = current_time
                self.last_speech_time = current_time
            elif self.is_speaking and (current_time - self.last_speech_time) > self.min_silence_duration_s:
                self.is_speaking = False
        except Exception as e:
            print(f"❌ Error in process_frame: {e}")

    def get_continuous_speech_info(self) -> Dict:
        if self.is_speaking:
            duration = time.time() - self.speech_start_time
            return {
                'is_speaking': True,
                'duration': duration,
                'start_time': self.speech_start_time
            }
        return {'is_speaking': False, 'duration': 0, 'start_time': None}

    def reset_speech_state(self):
        self.speech_buffer.clear()
        self.is_speaking = False
        self.last_speech_time = 0.0
        self.speech_start_time = 0.0

if __name__ == "__main__":
    vad = VoiceActivityDetector(sampling_rate=16000, threshold=0.5, min_speech_duration_ms=100)
    from audio.mic_stream import MicStream
    mic_stream = MicStream(samplerate=16000, channels=1, frame_duration_ms=100, hop_duration_ms=30)
    print("🎤 Starting continuous VAD test...")
    print("Speak long sentences to test (Press Ctrl+C to stop)")
    print("=" * 50)
    try:
        frame_count = 0
        for audio_frame in mic_stream.stream():
            frame_count += 1
            vad.process_frame(audio_frame)
            speech_info = vad.get_continuous_speech_info()
            audio_level = np.abs(audio_frame).mean()
            if frame_count % 5 == 0:
                if speech_info['is_speaking']:
                    print(f"🎤 SPEAKING ({speech_info['duration']:.1f}s) | Audio level: {audio_level:.4f}")
                else:
                    print(f"🔇 SILENT      | Audio level: {audio_level:.4f}")
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("🛑 Stopping VAD test")
    except Exception as e:
        print(f"❌ Error: {e}")
