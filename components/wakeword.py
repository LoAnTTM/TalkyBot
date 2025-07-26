import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import sounddevice as sd
from openwakeword import utils
from openwakeword.model import Model
from collections import deque

from components.logger import get_logger

class WakeWordDetector:
    def __init__(
        self,
        model_name="alexa",
        model_folder="models/openwakeword",
        onnx_file="alexa_v0.1.onnx",
        sensitivity_threshold=0.2,
        min_trigger_interval=1.0,
        state_manager=None,
    ):
        self.model_name = model_name
        self.model_folder = model_folder
        self.model_path = os.path.abspath(os.path.join(model_folder, onnx_file))
        self.sensitivity_threshold = sensitivity_threshold
        self.min_trigger_interval = min_trigger_interval
        self.last_trigger_time = 0.0
        self.state_manager = state_manager
        self.logger = get_logger("WAKE")

        self._prepare_model()

        self.model = Model(
            wakeword_models=[self.model_path],
            inference_framework="onnx",
        )
        # Sliding window buffer for 1s audio (16kHz)
        self.buffer_size = 16000  # 1s at 16kHz
        self.audio_buffer = deque(maxlen=self.buffer_size)

    def _prepare_model(self):
        os.makedirs(self.model_folder, exist_ok=True)
        if not os.path.isfile(self.model_path):
            print(f"Loading model '{self.model_name}'...")
            utils.download_models(
                model_names=[self.model_name],
                target_directory=self.model_folder
            )
        else:
            print(f"Model is already available at: {self.model_path}")

    # Get audio from vad and process it
    def process_frame(self, frame: np.ndarray):
        # print("WakeWordDetector: Received frame for wake word detection")
        try:
            # Convert float32 [-1, 1] to int16
            if not isinstance(frame, np.ndarray) or frame.ndim != 2:
                self.logger.warning("Invalid audio frame format for wake word detection")
                return
            
            audio_int16 = (frame[:, 0] * 32767).astype(np.int16)
            self.audio_buffer.extend(audio_int16.tolist())

            # if len(self.audio_buffer) < self.buffer_size:
            #     # Not enough audio for detection
            #     return
            
            # Convert deque to numpy array
            buffer_np = np.array(self.audio_buffer, dtype=np.int16)
            scores = self.model.predict(buffer_np)
            current_time = time.time()

            for wake_word, score in scores.items():
                self.logger.info(f"WakeWordDetector: score={score:.3f}")
                if (score > self.sensitivity_threshold and
                    (current_time - self.last_trigger_time > self.min_trigger_interval)):
                    if self.state_manager and hasattr(self.state_manager, 'wake_up'):
                        # Notify state manager to wake up
                        self.logger.info(f"Wake word '{wake_word}' detected!!!!! (score: {score:.3f})")
                    
                    self.last_trigger_time = current_time

        except Exception as e:
            self.logger.error(f"Error in wake word frame: {e}")


    # ========================================================
    # TODO: Remove this function after testing
    # Callback function for the audio stream from microphone
    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")

        audio_int16 = (indata[:, 0] * 32767).astype(np.int16)

        try:
            scores = self.model.predict(audio_int16)
            current_time = time.time()
            for wake_word, score in scores.items():
                if score > self.sensitivity_threshold and (current_time - self.last_trigger_time > self.min_trigger_interval):
                    print(f"Wake word '{wake_word}' detected! (score: {score:.3f})")
                    self.last_trigger_time = current_time
        except Exception as e:
            print(f"Error predicting wake word: {e}")

    def start_listening(self):
        print("Starting to listen... (say 'Alexa')")
        try:
            with sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype="float32",
                blocksize=512,
                callback=self._callback,
            ):
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping listening.")

    def cleanup(self):
        """Clean up wake word detector"""
        try:
            if hasattr(self.model, 'cleanup'):
                self.model.cleanup()
                print("Model released.")
        except Exception as e:
            print(f"Model cleanup error (ignored): {e}")

if __name__ == "__main__":
    detector = WakeWordDetector()
    detector.start_listening()
