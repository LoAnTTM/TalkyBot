import time
from components.state_manager import StateManager
from thread.thread_wakeup import WakeWordThread
from thread.thread_vad import VADThread

if __name__ == "__main__":
    state_manager = StateManager()
    wakeword_thread = WakeWordThread(state_manager=state_manager)
    vad_thread = VADThread(state_manager=state_manager, wakeword_detector=wakeword_thread)

    wakeword_thread.start()
    vad_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping threads...")
        vad_thread.stop()
        wakeword_thread.stop()
        vad_thread.join()
        wakeword_thread.join()
        print("Threads stopped.")