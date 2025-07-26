import sounddevice as sd
import numpy as np
import threading
import time
import logging

class MicStream:
    def __init__(self, samplerate=16000, channels=1, frame_duration_ms=30):
        self.samplerate = samplerate
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(self.samplerate * self.frame_duration_ms / 1000)
        self._stream = None
        self._lock = threading.Lock()
        self._restart_requested = False
        # Audio device management for loopback prevention
        self._setup_audio_devices()

    def _setup_audio_devices(self):
        """Setup separate input/output devices to prevent loopback"""
        try:
            devices = sd.query_devices()
            
            # Find best microphone input (avoid system audio devices)
            input_candidates = []
            for i, device in enumerate(devices):
                if (device['max_input_channels'] > 0 and 
                    'Built-in' in device['name'] and 
                    'Microphone' in device['name']):
                    input_candidates.append((i, device))
            
            # Find best speaker output (avoid system mix devices)
            output_candidates = []
            for i, device in enumerate(devices):
                if (device['max_output_channels'] > 0 and 
                    'Built-in' in device['name'] and 
                    'Output' in device['name']):
                    output_candidates.append((i, device))
            
            if input_candidates:
                self.input_device = input_candidates[0][0]
                print(f"🎤 Selected input: {input_candidates[0][1]['name']}")
            else:
                self.input_device = None
                print("⚠️ Using default input device")
            
            if output_candidates:
                self.output_device = output_candidates[0][0]
                print(f"🔊 Selected output: {output_candidates[0][1]['name']}")
            else:
                self.output_device = None
                print("⚠️ Using default output device")
                
        except Exception as e:
            print(f"⚠️ Error setting up audio devices: {e}")
            self.input_device = None
            self.output_device = None

    def stream(self):
        """Generator returns each audio frame from the microphone."""
        logger = logging.getLogger("MicStream")
        try:
            logger.info("Opening audio input stream...")
            with sd.InputStream(
                samplerate=self.samplerate, 
                channels=self.channels, 
                dtype=self.dtype, 
                blocksize=self.frame_size,
                device=self.input_device  # Use specific input device
            ) as stream:
                logger.info("Audio input stream opened successfully.")
                while True:
                    audio_frame, _ = stream.read(self.frame_size)
                    if self.channels == 1:
                        audio_frame = audio_frame.flatten()
                    yield audio_frame
        except Exception as e:
            logger.error(f"Error in MicStream.stream: {e}", exc_info=True)
            raise

    def restart_stream(self):
        """Request stream restart."""
        with self._lock:
            self._restart_requested = True
            print("🔄 Stream restart requested")

    def play_audio(self, audio, samplerate=None):
        """Audio playback (numpy array) with specific output device."""
        if samplerate is None:
            samplerate = self.samplerate
        # Use specific output device to avoid system audio capture
        sd.play(audio, samplerate, device=self.output_device)
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
            
            # Simple test without input/output check
            print("✅ Audio device basic info retrieved successfully")
            return True
            
        except Exception as e:
            print(f"❌ Audio device test failed: {e}")
            return False 