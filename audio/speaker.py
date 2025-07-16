import sounddevice as sd
from components.tts import TextToSpeech


class Speaker:
    def __init__(self, device="cpu"):
        self.tts = TextToSpeech(device=device)

    def speak(self, text):
        """Generate and play audio"""
        audio = self.tts.generate_audio(text)
        self.play_audio(audio)
        return audio

    def play_audio(self, audio):
        """Play audio"""
        try:
            sample_rate = 22050
            print(f"🔊 Playing audio ({len(audio)} samples at {sample_rate}Hz)...")
            sd.play(audio, samplerate=sample_rate)
            sd.wait()
            print("⏹️ Playback finished.")
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
