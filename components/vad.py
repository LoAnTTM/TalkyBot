import torch
import numpy as np
import time

class VoiceActivityDetector:
    def __init__(self, sampling_rate=16000, threshold=0.9, min_speech_duration_ms=250):
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        
        print("Loading Silero VAD model from torch hub...")
        try:
            # Load Silero VAD model directly from torch hub
            self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
            self.get_speech_timestamps, _, _, _, _ = utils
            print("VAD model loaded successfully!")
        except Exception as e:
            print(f"Error loading VAD model: {e}")
            raise
        
        # State to track continuous speech
        self.speech_buffer = []
        self.is_speaking = False
        self.last_speech_time = time.time()
        self.speech_start_time = None
        
        # Buffer size for continuous detection
        self.buffer_duration_ms = 2000  # 2 second buffer
        self.buffer_size = int(self.sampling_rate * self.buffer_duration_ms / 1000)

    def is_speech(self, audio_frame):
        # Ensure audio_frame is numpy array float32
        if not isinstance(audio_frame, np.ndarray):
            audio_frame = np.array(audio_frame)
        
        # Convert data type if needed
        if audio_frame.dtype != np.float32:
            if audio_frame.dtype == np.int16:
                audio_frame = audio_frame.astype(np.float32) / 32768.0
            else:
                audio_frame = audio_frame.astype(np.float32)
        
        # Ensure it's 1D array
        if audio_frame.ndim > 1:
            audio_frame = audio_frame.flatten()
        
        # Clean data - remove any invalid values
        audio_frame = np.nan_to_num(audio_frame, nan=0.0, posinf=0.0, neginf=0.0)
        audio_frame = np.clip(audio_frame, -1.0, 1.0)
        
        # Add to speech buffer
        self.speech_buffer.extend(audio_frame)
        
        # Keep buffer within desired size
        if len(self.speech_buffer) > self.buffer_size:
            self.speech_buffer = self.speech_buffer[-self.buffer_size:]
        
        # Check minimum length
        if len(self.speech_buffer) < 512:
            return False
        
        # Create numpy array from buffer
        buffer_array = np.array(self.speech_buffer, dtype=np.float32)
        
        # Use get_speech_timestamps to detect speech
        speech_timestamps = self.get_speech_timestamps(buffer_array, self.model, sampling_rate=self.sampling_rate)
        
        current_time = time.time()
        
        # Return True if there are any speech segments
        has_speech = len(speech_timestamps) > 0
        
        if has_speech:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = current_time
            self.last_speech_time = current_time
        else:
            # Check if stopped speaking (silent for 0.5 seconds)
            if self.is_speaking and (current_time - self.last_speech_time) > 0.5:
                self.is_speaking = False
                self.speech_start_time = None
        
        return has_speech or self.is_speaking
    
    def get_speech_segments(self, audio_frame):
        """Return speech segments with timestamps"""
        if not isinstance(audio_frame, np.ndarray):
            audio_frame = np.array(audio_frame)
        
        if audio_frame.dtype != np.float32:
            if audio_frame.dtype == np.int16:
                audio_frame = audio_frame.astype(np.float32) / 32768.0
            else:
                audio_frame = audio_frame.astype(np.float32)
        
        if audio_frame.ndim > 1:
            audio_frame = audio_frame.flatten()
            
        if len(audio_frame) < 512:
            return []
            
        return self.get_speech_timestamps(audio_frame, self.model, sampling_rate=self.sampling_rate)
    
    def get_continuous_speech_info(self):
        """Return information about current continuous speech session"""
        if self.is_speaking and self.speech_start_time:
            duration = time.time() - self.speech_start_time
            return {
                'is_speaking': True,
                'duration': duration,
                'start_time': self.speech_start_time
            }
        return {
            'is_speaking': False,
            'duration': 0,
            'start_time': None
        }
    
    def reset_speech_state(self):
        """Reset speech detection state"""
        self.speech_buffer = []
        self.is_speaking = False
        self.last_speech_time = time.time()
        self.speech_start_time = None