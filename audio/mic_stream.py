import sounddevice as sd
import numpy as np

class AudioStream:
    def __init__(self, samplerate=16000, channels=1, frame_duration_ms=30):
        self.samplerate = samplerate
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(self.samplerate * self.frame_duration_ms / 1000)

    def stream(self):
        """Generator returns each audio frame from the microphone."""
        with sd.InputStream(samplerate=self.samplerate, channels=self.channels, dtype='int16', blocksize=self.frame_size) as stream:
            while True:
                audio_frame, _ = stream.read(self.frame_size)
                # Convert to 1D if only 1 channel
                if self.channels == 1:
                    audio_frame = audio_frame.flatten()
                yield audio_frame

    def play_audio(self, audio, samplerate=None):
        """Audio playback (numpy array)."""
        if samplerate is None:
            samplerate = self.samplerate
        sd.play(audio, samplerate)
        sd.wait() 