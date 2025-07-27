import time
from components.state_manager import StateManager
from thread.thread_wakeup import WakeWordThread
from thread.thread_vad import VADThread
from thread.thread_stt import STTConversationThread
from thread.thread_tts import TTSThread

if __name__ == "__main__":
    state_manager = StateManager()

    wakeword_thread = WakeWordThread(state_manager=state_manager)
    tts_thread = TTSThread(state_manager=state_manager)
    stt_thread = STTConversationThread(state_manager=state_manager, response_callback=tts_thread.speak_text)
    vad_thread = VADThread(state_manager=state_manager, wakeword_detector=wakeword_thread, stt_handler=stt_thread)

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