import numpy as np
import sounddevice as sd
from TTS.api import TTS
import os


class TextToSpeech:
    def __init__(self, model_name="tts_models/en/ljspeech/vits"):
        if not model_name:
            raise ValueError("Model name cannot be empty.")

        # Ensure model exists (downloads if needed)
        self.model_name = model_name
        self.tts = TTS(model_name=self.model_name)

        # Hide unnecessary output during initialization
        # with contextlib.redirect_stdout(io.StringIO()):
        #     self.tts = TTS(model_name=self.model_name)

        # Confirm synthesizer loaded properly
        if not hasattr(self.tts, "synthesizer"):
            raise RuntimeError("Failed to load TTS synthesizer.")

        self.sample_rate = self.tts.synthesizer.output_sample_rate

    def speak(self, text):
        if not text:
            raise ValueError("Text input cannot be empty.")

        #hide unnecessary output during synthesis
        # with contextlib.redirect_stdout(io.StringIO()):
        # wav = self.tts.tts(text)

        wav = self.tts.tts(text)
        wav = wav / np.abs(wav).max()  # Normalize audio
        sd.play(wav, samplerate=self.sample_rate)
        sd.wait()


if __name__ == "__main__":
    tts = TextToSpeech()
    tts.speak("Two antennas met on a roof, fell in love, and got married. The ceremony wasn't much, but the reception was excellent.")
