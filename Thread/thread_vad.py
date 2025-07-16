import threading
import time

from components.vad import VoiceActivityDetector

class VADThread(threading.Thread):
    def __init__(self, vad, audio_stream, audio_queue=None, status_callback=None):
        super().__init__()
        self.vad = vad
        self.audio_stream = audio_stream
        self.audio_queue = audio_queue  # Queue để gửi audio frames cho STT
        self.status_callback = status_callback
        self._stop_event = threading.Event()
        self.last_speaking_state = None
        self.frame_count = 0

    def run(self):
        while not self._stop_event.is_set():
            for frame in self.audio_stream.stream():
                if self._stop_event.is_set():
                    break
                self.frame_count += 1
                self.vad.process_frame(frame)
                info = self.vad.get_continuous_speech_info()
                
                # Gửi audio frame cho STT khi có speech
                if info['is_speaking'] and self.audio_queue:
                    self.audio_queue.put(frame)
                
                # Only call callback when state changes or every 10 frames during speech
                current_speaking = info['is_speaking']
                if (current_speaking != self.last_speaking_state or 
                    (current_speaking and self.frame_count % 10 == 0)):
                    if self.status_callback:
                        self.status_callback(info)
                    self.last_speaking_state = current_speaking
                
                time.sleep(0.01)

    def stop(self):
        self._stop_event.set()


def vad_status_callback(info):
    """Callback function to handle VAD status updates with reduced spam"""
    if info['is_speaking']:
        print(f"🎤 Speech detected - Duration: {info['duration']:.1f}s")
    else:
        print("🔇 Speech ended - Silence detected")

if __name__ == "__main__":
    from audio.mic_stream import AudioStream
    
    # Create VAD detector and audio stream
    vad = VoiceActivityDetector(sampling_rate=16000, threshold=0.5, min_speech_duration_ms=100)
    audio_stream = AudioStream(samplerate=16000, channels=1, frame_duration_ms=100)
    
    # Create VAD thread with callback to receive status
    vad_thread = VADThread(vad, audio_stream, status_callback=vad_status_callback)
    
    # Start the thread
    vad_thread.start()
    
    print("🎤 VAD Thread test started!")
    print("Speak to test speech detection (Press Ctrl+C to stop)")
    print("=" * 50)

    try:
        # Main thread can do other work or just wait
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("🛑 Stopping program...")
        vad_thread.stop()
        vad_thread.join()
        print("✅ Program stopped.")