from vosk import Model, KaldiRecognizer
import numpy as np
import json

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from audio.mic_stream import AudioStream

class SpeechToText:
    def __init__(self, model_path="models/vosk/vosk-model-small-en-us-0.15", samplerate=16000):
        if not os.path.exists(model_path):
            raise RuntimeError(f"❌ Vosk model not found at {model_path}. Please download from https://alphacephei.com/vosk/models")
        self.model = Model(model_path)
        self.samplerate = samplerate

    def _to_int16(self, audio):
        if audio.dtype != np.int16:
            return (audio * 32768).astype(np.int16)
        return audio

    def transcribe(self, audio):
        audio = self._to_int16(audio)
        rec = KaldiRecognizer(self.model, self.samplerate)
        rec.AcceptWaveform(audio.tobytes())
        result = json.loads(rec.Result())
        return result.get("text", "").strip()

    def transcribe_continuous(self, audio):
        audio = self._to_int16(audio)
        rec = KaldiRecognizer(self.model, self.samplerate)
        if rec.AcceptWaveform(audio.tobytes()):
            result = json.loads(rec.Result())
            return result.get("text", "").strip(), True
        else:
            partial = json.loads(rec.PartialResult())
            return partial.get("partial", "").strip(), False

    def listen_and_transcribe(self, frame_duration_ms=250):
        """Real-time STT loop with print output"""
        print("🎙️ Listening from STT...")
        audio_stream = AudioStream(samplerate=self.samplerate, channels=1, frame_duration_ms=frame_duration_ms)
        rec = KaldiRecognizer(self.model, self.samplerate)
        current_partial = ""

        try:
            for frame in audio_stream.stream():
                frame = self._to_int16(frame)
                if rec.AcceptWaveform(frame.tobytes()):
                    result = json.loads(rec.Result()).get("text", "").strip()
                    if result:
                        print(f"\r{' '*80}\r✅ Final: '{result}'")
                    current_partial = ""
                else:
                    partial = json.loads(rec.PartialResult()).get("partial", "").strip()
                    if partial and partial != current_partial:
                        print(f"\r{' '*80}\r🔄 Partial: '{partial}'", end='', flush=True)
                        current_partial = partial
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user.")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    stt = SpeechToText()
    stt.listen_and_transcribe()