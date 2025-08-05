import sounddevice as sd
import numpy as np
from collections import deque

class MicStream:
    def __init__(self, samplerate=16000, channels=1, frame_duration_ms=30, hop_duration_ms=10):
        self.samplerate = samplerate
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        self.hop_duration_ms = hop_duration_ms

        self.frame_size = int(self.samplerate * self.frame_duration_ms / 1000)
        self.hop_size = int(self.samplerate * self.hop_duration_ms / 1000)

        # Use a deque as a sliding window buffer
        self._buffer = deque(maxlen=self.frame_size)

    def stream(self):
        """Generator that yields overlapping audio frames from the microphone."""
        # The blocksize should be the hop_size to get new data in smaller chunks
        with sd.InputStream(samplerate=self.samplerate, channels=self.channels, dtype='int16', blocksize=self.hop_size) as stream:
            # Pre-fill the buffer before starting to yield frames
            initial_data, _ = stream.read(self.frame_size - self.hop_size)
            if self.channels == 1:
                initial_data = initial_data.flatten()
            self._buffer.extend(initial_data)

            while True:
                # Read a new chunk of data (the hop size)
                new_chunk, _ = stream.read(self.hop_size)
                if self.channels == 1:
                    new_chunk = new_chunk.flatten()

                # The deque will automatically slide the window
                self._buffer.extend(new_chunk)

                # Yield the current window (a copy of the buffer)
                yield np.array(self._buffer, dtype=np.int16)