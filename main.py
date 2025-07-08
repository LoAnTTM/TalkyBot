from audio.mic_stream import AudioStream
from voice.vad import VoiceActivityDetector
from wake.wakeword import WakeWordDetector
from audio.recorder import SpeechRecorder
from stt.vosk_stt import SpeechToText
from nlp.dialogpt import Chatbot
from tts.coqui_tts import TextToSpeech

def main():
    # 1. Initialize modules
    mic = AudioStream()
    vad = VoiceActivityDetector()
    wakeword = WakeWordDetector()
    recorder = SpeechRecorder()
    stt = SpeechToText()
    nlp = Chatbot()
    tts = TextToSpeech()

    # Track wake word detection state
    wake_word_detected = False
    
    def on_wake_word_detected():
        nonlocal wake_word_detected
        wake_word_detected = True
        print("🎤 Wake word detected! Listening for command...")
    
    # Set up wake word callback
    wakeword.set_wake_callback(on_wake_word_detected)

    print("TalkyBot is listening...")

    # 2. Main loop
    for audio_frame in mic.stream():
        if not vad.is_speech(audio_frame):
            continue

        # Process wake word detection
        wakeword.detect_from_voice(audio_frame)
        
        # If wake word was detected, start recording speech
        if wake_word_detected:
            wake_word_detected = False  # Reset flag
            
            speech_audio = recorder.record(mic, vad)
            # Properly check if speech_audio is None or empty
            if speech_audio is None or len(speech_audio) == 0:
                print("No speech recorded")
                continue

            text = stt.transcribe(speech_audio)
            if not text:
                print("No text transcribed")
                continue

            print("User:", text)
            response = nlp.get_response(text)
            print("Bot:", response)

            tts.speak(response)

if __name__ == "__main__":
    main() 