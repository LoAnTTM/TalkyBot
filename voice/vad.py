import torch
import numpy as np
import time

class VoiceActivityDetector:
    def __init__(self, sampling_rate=16000, threshold=0.5, min_speech_duration_ms=250):
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        
        # Load Silero VAD model trực tiếp
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        self.get_speech_timestamps, _, _, _, _ = utils
        
        # Trạng thái để theo dõi speech liên tục
        self.speech_buffer = []
        self.is_speaking = False
        self.last_speech_time = time.time()
        self.speech_start_time = None
        
        # Buffer size for continuous detection
        self.buffer_duration_ms = 2000  # 2 giây buffer
        self.buffer_size = int(self.sampling_rate * self.buffer_duration_ms / 1000)

    def is_speech(self, audio_frame):
        # Đảm bảo audio_frame là numpy array float32
        if not isinstance(audio_frame, np.ndarray):
            audio_frame = np.array(audio_frame)
        
        # Chuyển đổi kiểu dữ liệu nếu cần
        if audio_frame.dtype != np.float32:
            if audio_frame.dtype == np.int16:
                audio_frame = audio_frame.astype(np.float32) / 32768.0
            else:
                audio_frame = audio_frame.astype(np.float32)
        
        # Đảm bảo là 1D array
        if audio_frame.ndim > 1:
            audio_frame = audio_frame.flatten()
        
        # Thêm vào speech buffer
        self.speech_buffer.extend(audio_frame)
        
        # Giữ buffer trong kích thước mong muốn
        if len(self.speech_buffer) > self.buffer_size:
            self.speech_buffer = self.speech_buffer[-self.buffer_size:]
        
        # Kiểm tra độ dài tối thiểu
        if len(self.speech_buffer) < 512:
            return False
        
        # Tạo numpy array từ buffer
        buffer_array = np.array(self.speech_buffer, dtype=np.float32)
        
        # Sử dụng get_speech_timestamps để phát hiện tiếng nói
        speech_timestamps = self.get_speech_timestamps(buffer_array, self.model, sampling_rate=self.sampling_rate)
        
        current_time = time.time()
        
        # Trả về True nếu có bất kỳ speech segment nào
        has_speech = len(speech_timestamps) > 0
        
        if has_speech:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = current_time
            self.last_speech_time = current_time
        else:
            # Kiểm tra xem có ngừng nói không (im lặng trong 0.5 giây)
            if self.is_speaking and (current_time - self.last_speech_time) > 0.5:
                self.is_speaking = False
                self.speech_start_time = None
        
        return has_speech or self.is_speaking
    
    def get_speech_segments(self, audio_frame):
        """Trả về các segment có tiếng nói với timestamp"""
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
        """Trả về thông tin về phiên nói liên tục hiện tại"""
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
        """Reset trạng thái phát hiện speech"""
        self.speech_buffer = []
        self.is_speaking = False
        self.last_speech_time = time.time()
        self.speech_start_time = None