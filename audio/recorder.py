import numpy as np
import time

class SpeechRecorder:
    def __init__(self, max_silence_sec=1.0, max_record_sec=10):
        self.max_silence_sec = max_silence_sec
        self.max_record_sec = max_record_sec

    def record(self, mic, vad):
        """
        Record speech after wake word.
        mic: AudioStream instance
        vad: VoiceActivityDetector instance
        Returns recorded audio numpy array.
        """
        audio_frames = []
        silence_start = None
        start_time = time.time()
        for audio_frame in mic.stream():
            if vad.is_speech(audio_frame):
                audio_frames.append(audio_frame)
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > self.max_silence_sec:
                    break
            # Stop if exceeding maximum time
            if time.time() - start_time > self.max_record_sec:
                break
        if audio_frames:
            return np.concatenate(audio_frames)
        return None 