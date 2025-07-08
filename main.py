from audio.mic_stream import AudioStream
from voice.vad import VoiceActivityDetector
from wake.wakeword import WakeWordDetector
from audio.recorder import SpeechRecorder
from stt.vosk_stt import SpeechToText
from nlp.dialogpt import Chatbot
from tts.coqui_tts import TextToSpeech

def main():
    # 1. Khởi tạo các module
    mic = AudioStream()
    vad = VoiceActivityDetector()
    wakeword = WakeWordDetector()
    recorder = SpeechRecorder()
    stt = SpeechToText()
    nlp = Chatbot()
    tts = TextToSpeech()

    print("TalkyBot is listening...")

    # 2. Vòng lặp chính
    for audio_frame in mic.stream():
        if not vad.is_speech(audio_frame):
            continue

        if not wakeword.detect(audio_frame):
            continue

        print("Wake word detected! Listening for command...")

        speech_audio = recorder.record(mic, vad)
        if not speech_audio:
            continue

        text = stt.transcribe(speech_audio)
        if not text:
            continue

        print("User:", text)
        response = nlp.get_response(text)
        print("Bot:", response)

        response_audio = tts.speak(response)
        mic.play_audio(response_audio)

if __name__ == "__main__":
    main() 