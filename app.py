import time
from components.state_manager import StateManager
from thread.thread_wakeup import WakeWordThread
from thread.thread_vad import VADThread
from thread.thread_stt import STTConversationThread

if __name__ == "__main__":
    state_manager = StateManager()

    wakeword_thread = WakeWordThread(state_manager=state_manager)
    stt_handler = STTConversationThread(state_manager=state_manager)
    vad_thread = VADThread(state_manager=state_manager, wakeword_detector=wakeword_thread, stt_handler=stt_handler)

    wakeword_thread.start()
    stt_handler.start()
    vad_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping threads...")
        vad_thread.stop()
        wakeword_thread.stop()
        stt_handler.stop()
        vad_thread.join()
        wakeword_thread.join()
        stt_handler.join()
        print("Threads stopped.")