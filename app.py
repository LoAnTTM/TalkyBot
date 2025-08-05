import time
import threading
from audio.mic_stream import MicStream
from components.state_manager import StateManager
from thread.thread_wakeup import WakeWordThread
from thread.thread_vad import VADThread
from thread.thread_stt import STTConversationThread
from thread.thread_tts import TTSThread

# Shared resources
mic_stream = MicStream(samplerate=16000, channels=1, frame_duration_ms=1000, hop_duration_ms=30)
state_manager = StateManager()
tts_interrupt_event = threading.Event()
tts_playing_event = threading.Event()

# Initialize threads
wakeword_thread = WakeWordThread(state_manager=state_manager)
tts_thread = TTSThread(state_manager=state_manager, interrupt_event=tts_interrupt_event, playing_event=tts_playing_event)
stt_thread = STTConversationThread(state_manager=state_manager, response_callback=tts_thread.speak_text)
vad_thread = VADThread(mic_stream=mic_stream, 
                     state_manager=state_manager, 
                     wakeword_thread=wakeword_thread, 
                     stt_thread=stt_thread, 
                     tts_interrupt_event=tts_interrupt_event,
                     tts_playing_event=tts_playing_event)

wakeword_thread.start()
stt_thread.start()
vad_thread.start()
tts_thread.start()

try:
    while True:
        time.sleep(1)
        state_manager.check_timeout()
except KeyboardInterrupt:
    print("Stopping threads...")
    vad_thread.stop()
    wakeword_thread.stop()
    stt_thread.stop()
    tts_thread.stop()
    vad_thread.join()
    wakeword_thread.join()
    stt_thread.join()
    tts_thread.join()
    print("Threads stopped.")