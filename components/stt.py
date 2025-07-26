import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from audio.mic_stream import MicStream

from vosk import Model, KaldiRecognizer
import numpy as np
import json


class SpeechToText:
    def __init__(self, model_path="models/vosk/vosk-model-small-en-us-0.15", samplerate=16000):
        if not os.path.exists(model_path):
            raise RuntimeError(f"❌ Vosk model not found at {model_path}. Please download from https://alphacephei.com/vosk/models")
        self.model = Model(model_path)

         # Suppress Vosk C++ logs
        # sys_stdout = sys.stdout
        # sys_stderr = sys.stderr
        # try:
        #     sys.stdout = open(os.devnull, 'w')
        #     sys.stderr = open(os.devnull, 'w')
        #     self.model = Model(model_path)
        # finally:
        #     sys.stdout = sys_stdout
        #     sys.stderr = sys_stderr

        self.samplerate = samplerate

        self.rec = KaldiRecognizer(self.model, self.samplerate)
        self.current_partial = ""

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

    def process_frame(self, frame: np.ndarray):
        frame = self._to_int16(frame)
        if self.rec.AcceptWaveform(frame.tobytes()):
            result = json.loads(self.rec.Result()).get("text", "").strip()
            if result:
                print(f"\r{' '*80}\r✅ Final: '{result}'")
                return result
            self.current_partial = ""
        else:
            partial = json.loads(self.rec.PartialResult()).get("partial", "").strip()
            if partial and partial != self.current_partial:
                print(f"\r{' '*80}\r🔄 Partial: '{partial}'", end='', flush=True)
                self.current_partial = partial
        return None

    def listen_and_transcribe(self, frame_duration_ms=250):
        """Real-time STT loop with print output"""
        print("🎙️ Listening from STT...")
        mic_stream = MicStream(samplerate=self.samplerate, channels=1, frame_duration_ms=frame_duration_ms)
        rec = KaldiRecognizer(self.model, self.samplerate)
        current_partial = ""

        try:
            for frame in mic_stream.stream():
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