import sounddevice as sd
import numpy as np
import threading
import time
import logging
from collections import deque

# Configure module-level logger to write into TalkyBot.log
logger = logging.getLogger("MicStream")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("TalkyBot.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class MicStream:
    """
    Manage microphone input with a sliding-window buffer for real-time audio processing.
    """
    def __init__(
        self,
        samplerate: int = 16000,
        channels: int = 1,
        frame_duration_ms: int = 30,
        hop_duration_ms: int = 10,
        buffer_duration_s: float = 10.0
    ):
        # Audio parameters
        self.samplerate = samplerate
        self.channels = channels
        self.frame_size = int(self.samplerate * frame_duration_ms / 1000)
        self.hop_size = int(self.samplerate * hop_duration_ms / 1000)
        self.buffer_size = int(self.samplerate * buffer_duration_s)

        # Thread-safe circular buffer
        self._buffer = deque(maxlen=self.buffer_size)
        self._lock = threading.Lock()

        # Sounddevice stream
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="int16",
            callback=self._audio_callback
        )
        self._running = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback from sounddevice for each incoming block."""
        if status:
            logger.warning("Audio status: %s", status)
        # indata shape: (frames, channels)
        samples = indata[:, 0]  # take first channel
        with self._lock:
            self._buffer.extend(samples)

    def start(self):
        """Start capturing audio from microphone."""
        if not self._running:
            self._stream.start()
            self._running = True
            logger.info("Microphone stream started.")

    def stop(self):
        """Stop capturing audio and clear buffer."""
        if self._running:
            self._stream.stop()
            self._running = False
            with self._lock:
                self._buffer.clear()
            logger.info("Microphone stream stopped and buffer cleared.")

    def stream_frames(self):
        """
        Generator that yields sliding-window frames continuously.
        Each frame has length = frame_size, windows slide by hop_size.
        """
        try:
            while self._running:
                window = None
                with self._lock:
                    if len(self._buffer) >= self.frame_size:
                        # Extract frame_size samples
                        frame = [self._buffer[i] for i in range(self.frame_size)]
                        # Drop hop_size samples
                        for _ in range(self.hop_size):
                            self._buffer.popleft()
                        window = np.array(frame, dtype=np.int16)
                if window is None:
                    # Not enough data yet, wait a bit
                    time.sleep(self.hop_size / self.samplerate)
                else:
                    yield window
        except Exception as e:
            logger.error("Error in stream_frames: %s", e, exc_info=True)

    def close(self):
        """Close audio stream and release resources."""
        if self._stream:
            self._stream.close()
            logger.info("Microphone stream closed.")
