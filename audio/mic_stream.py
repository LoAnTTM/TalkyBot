import sounddevice as sd
import numpy as np
import threading
import time
import logging

class AudioStream:
    def __init__(self, samplerate=16000, channels=1, frame_duration_ms=30):
        self.samplerate = samplerate
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(self.samplerate * self.frame_duration_ms / 1000)
        self._stream = None
        self._lock = threading.Lock()
        self._restart_requested = False

    def stream(self):
        """Generator returns each audio frame from the microphone."""
        logger = logging.getLogger("AudioStream")
        try:
            logger.info("Opening audio input stream...")
            with sd.InputStream(samplerate=self.samplerate, channels=self.channels, dtype='int16', blocksize=self.frame_size) as stream:
                logger.info("Audio input stream opened successfully.")
                while True:
                    audio_frame, _ = stream.read(self.frame_size)
                    if self.channels == 1:
                        audio_frame = audio_frame.flatten()
                    yield audio_frame
        except Exception as e:
            logger.error(f"Error in AudioStream.stream: {e}", exc_info=True)
            raise

    def restart_stream(self):
        """Request stream restart."""
        with self._lock:
            self._restart_requested = True
            print("🔄 Stream restart requested")

    def play_audio(self, audio, samplerate=None):
        """Audio playback (numpy array)."""
        if samplerate is None:
            samplerate = self.samplerate
        sd.play(audio, samplerate)
        sd.wait() 

    def get_devices(self):
        """List available audio devices for debugging."""
        return sd.query_devices()

    def test_device(self):
        """Test if default audio device is working."""
        try:
            print("🔍 Testing default audio device...")
            device_info = sd.query_devices(sd.default.device)
            print(f"📱 Default device: {device_info['name']}")
            print(f"📊 Max input channels: {device_info['max_input_channels']}")
            print(f"🔊 Default samplerate: {device_info['default_samplerate']}")
            
            # Test recording a small sample
            duration = 0.1  # 100ms
            test_recording = sd.rec(
                int(duration * self.samplerate), 
                samplerate=self.samplerate, 
                channels=self.channels,
                dtype='int16'
            )
            sd.wait()
            
            # Check if we got audio data
            audio_level = np.abs(test_recording).mean()
            print(f"✅ Audio test successful - Level: {audio_level:.4f}")
            return True
            
        except Exception as e:
            print(f"❌ Audio device test failed: {e}")
            return False 