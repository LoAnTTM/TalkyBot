from vosk import Model, KaldiRecognizer
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from audio.mic_stream import AudioStream

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
    
    def transcribe_continuous(self, audio):
        """Transcribe audio continuously with partial results"""
        if audio.dtype != np.int16:
            audio = (audio * 32768).astype(np.int16)
        rec = KaldiRecognizer(self.model, self.samplerate)
        
        if rec.AcceptWaveform(audio.tobytes()):
            # Final result
            result = rec.Result()
            import json
            return json.loads(result).get("text", ""), True  # (text, is_final)
        else:
            # Partial result
            partial = rec.PartialResult()
            import json
            return json.loads(partial).get("partial", ""), False  # (text, is_final)

# Test function
if __name__ == "__main__":
    print("🎤 Testing Vosk Speech-to-Text...")
    
    # Kiểm tra model path
    model_path = "models/vosk/vosk-model-small-en-us-0.15"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please download Vosk model first!")
        print("Download from: https://alphacephei.com/vosk/models")
        sys.exit(1)
    
    try:
        # Khởi tạo STT
        stt = SpeechToText(model_path)
        print("✅ Vosk model loaded successfully!")
        
        # Khởi tạo audio stream với chunks nhỏ hơn cho real-time
        audio_stream = AudioStream(samplerate=16000, channels=1, frame_duration_ms=250)  # 250ms chunks
        print("🎙️  Listening... Speak continuously (Press Ctrl+C to stop)")
        print("📝 Real-time transcription:")
        
        # Khởi tạo recognizer cho continuous transcription
        rec = KaldiRecognizer(stt.model, stt.samplerate)
        current_sentence = ""
        
        for audio_frame in audio_stream.stream():
            # Chuyển đổi audio
            if audio_frame.dtype != np.int16:
                audio_frame = (audio_frame * 32768).astype(np.int16)
            
            # Transcribe continuous
            if rec.AcceptWaveform(audio_frame.tobytes()):
                # Final result - câu hoàn chỉnh
                result = rec.Result()
                import json
                final_text = json.loads(result).get("text", "").strip()
                if final_text:
                    # Xóa dòng partial và in final result
                    print(f"\r{' ' * 80}\r✅ Final: '{final_text}'")
                current_sentence = ""
            else:
                # Partial result - từ đang nói
                partial = rec.PartialResult()
                import json
                partial_text = json.loads(partial).get("partial", "").strip()
                if partial_text and partial_text != current_sentence:
                    # Xóa dòng hiện tại và in partial text
                    print(f"\r{' ' * 80}\r🔄 Partial: '{partial_text}'", end='', flush=True)
                    current_sentence = partial_text
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped listening.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure:")
        print("1. Vosk model is downloaded and extracted to the correct path")
        print("2. Microphone permissions are granted")
        print("3. All dependencies are installed") 