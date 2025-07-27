import threading
import queue
import asyncio

from components.tts import TextToSpeech
from components.logger import get_logger
from components.state_manager import SystemState

class TTSThread(threading.Thread):
    def __init__(self, state_manager=None, max_queue_size=10, 
                 interrupt_event=None, playing_event=None):
        super().__init__()
        self.state_manager = state_manager
        self.text_queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self.interrupt_event = interrupt_event or threading.Event()
        self.playing_event = playing_event or threading.Event()
        self.logger = get_logger("TTS")

        self.tts = TextToSpeech()
        self.loop = asyncio.new_event_loop()
        self.is_speaking = False

    def run(self):
        self.logger.info("TTS Thread started using async playback")
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self._run_async())  # run concurrently
        try:
            self.loop.run_forever()
        except Exception as e:
            self.logger.error(f"TTS loop error: {e}")
        finally:
            self.loop.close()
            self.logger.info("TTS event loop closed")

    async def _run_async(self):
        while not self._stop_event.is_set():
            try:
                if self.state_manager and not self.state_manager.conversation_active.is_set():
                    await asyncio.sleep(0.1)
                    continue

                text = await self.loop.run_in_executor(None, self._get_text)
                if text:
                    self.is_speaking = True
                    self.playing_event.set()
                    if self.state_manager:
                        self.state_manager.transition_to(SystemState.SPEAKING, "TTS speaking")
                    
                    # Reset interrupt
                    self.interrupt_event.clear()

                    await self.tts.speak(text, stop_event=self.interrupt_event)

                    self.playing_event.clear()
                    self.is_speaking = False
                    if self.state_manager:
                        self.state_manager.transition_to(SystemState.LISTENING, "TTS finished")

            except Exception as e:
                self.logger.error(f"TTS Error: {e}")
                self.playing_event.clear()
                self.is_speaking = False
                if self.state_manager:
                    self.state_manager.transition_to(SystemState.LISTENING, "TTS error")

    def _get_text(self):
        try:
            return self.text_queue.get(timeout=0.5)
        except queue.Empty:
            return None

    def speak_text(self, text, priority=False):
        if not text or not text.strip():
            return False
        if self.state_manager and not self.state_manager.conversation_active.is_set():
            return False

        try:
            if priority:
                self.clear_queue()
                if self.is_speaking:
                    self.interrupt_event.set()
            self.text_queue.put(text.strip(), timeout=0.1)
            return True
        except queue.Full:
            self.logger.warning("TTS queue is full, message dropped")
            return False

    def clear_queue(self):
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                break

    def interrupt_current_speech(self):
        self.interrupt_event.set()

    def stop(self):
        """Stop TTS thread safely"""
        self._stop_event.set()
        self.interrupt_event.set()
        
        def _shutdown_loop():
            if self.loop.is_running():
                for task in asyncio.all_tasks(self.loop):
                    task.cancel()
                self.loop.stop()
        
        self.loop.call_soon_threadsafe(_shutdown_loop)


    def get_queue_size(self):
        return self.text_queue.qsize()
