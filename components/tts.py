import numpy as np
import sounddevice as sd
import asyncio
from TTS.api import TTS

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class TextToSpeech:
    def __init__(self, model_name="tts_models/en/ljspeech/vits"):
        if not model_name:
            raise ValueError("Model name cannot be empty.")
        
        # Ensure model exists (downloads if needed)
        self.model_name = model_name
        self.tts = TTS(model_name=self.model_name)
        
        # Confirm synthesizer loaded properly
        if not hasattr(self.tts, "synthesizer"):
            raise RuntimeError("Failed to load TTS synthesizer.")
        self.sample_rate = self.tts.synthesizer.output_sample_rate

    def generate_audio(self, text):
        if not text:
            raise ValueError("Text input cannot be empty.")
        wav = self.tts.tts(text)
        return wav

    async def speak(self, text, stop_event: asyncio.Event = None):
        if not text:
            raise ValueError("Text input cannot be empty.")
        wav = self.generate_audio(text)
        wav = wav / np.abs(wav).max()
        wav = wav.astype(np.float32)

        # Play audio in small chunks, check stop_event
        blocksize = 2048
        idx = 0
        stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype='float32')
        stream.start()
        try:
            while idx < len(wav):
                if stop_event and stop_event.is_set():
                    print("🛑 TTS interrupted by VAD!")
                    sd.stop() 
                    stream.abort()
                    break
                end = min(idx + blocksize, len(wav))
                stream.write(wav[idx:end])
                idx = end
        except Exception as e:
            print(f"⚠️ Playback error: {e}")
        finally:
            stream.stop()
            stream.close()

    async def stop(self, stop_event: asyncio.Event):
        """Set stop event to interrupt playback"""
        stop_event.set()
        sd.stop()

if __name__ == "__main__":
    from thread.thread_vad import VADThread
    from components.state_manager import StateManager
    import threading
    import queue

    async def main():
        tts = TextToSpeech()
        stop_event = threading.Event()
        state_manager = StateManager()
        audio_queue = queue.Queue()

        # Khởi động VAD thread, truyền stop_event cho TTS interrupt
        vad_thread = VADThread(
            state_manager=state_manager,
            audio_queue=audio_queue,
            tts_interrupt_event=stop_event
        )
        vad_thread.start()

        # Phát TTS, sẽ bị ngắt nếu VAD phát hiện tiếng nói
        await tts.speak("Two antennas met on a roof, fell in love, and got married. The ceremony wasn't much, but the reception was excellent.", stop_event)

        # Dừng VAD thread sau khi test
        vad_thread.stop()
        vad_thread.join()


    asyncio.run(main())