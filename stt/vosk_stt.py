from vosk import Model, KaldiRecognizer
import numpy as np
import os

class SpeechToText:
    def __init__(self, model_path="models/vosk/vosk-model-small-en-us-0.15", samplerate=16000):
        if not os.path.exists(model_path):
            raise RuntimeError(f"Vosk model not found at {model_path}. Download and extract it!")
        self.model = Model(model_path)
        self.samplerate = samplerate

    def transcribe(self, audio):
        # Đảm bảo audio là int16 mono
        if audio.dtype != np.int16:
            audio = (audio * 32768).astype(np.int16)
        rec = KaldiRecognizer(self.model, self.samplerate)
        rec.AcceptWaveform(audio.tobytes())
        result = rec.Result()
        import json
        text = json.loads(result).get("text", "")
        return text.strip() if text else None 