import sounddevice as sd
from text_to_speech import TextToSpeech


class Speaker:
    def __init__(self, device="cpu"):
        self.tts = TextToSpeech(device=device)

    def speak(self, text):
        """Tạo và phát audio"""
        audio = self.tts.generate_audio(text)
        self.play_audio(audio)
        return audio

    def play_audio(self, audio):
        """Phát audio"""
        try:
            sample_rate = 22050
            print(f"🔊 Đang phát audio ({len(audio)} mẫu ở {sample_rate}Hz)...")
            sd.play(audio, samplerate=sample_rate)
            sd.wait()
            print("⏹️ Phát xong.")
        except Exception as e:
            print(f"❌ Lỗi phát audio: {e}")
