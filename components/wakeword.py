import os
import time
import numpy as np
import sounddevice as sd
from openwakeword import utils
from openwakeword.model import Model

class WakeWordDetector:
    def __init__(
        self,
        model_name="alexa",
        model_folder="models/openwakeword",
        onnx_file="alexa_v0.1.onnx",
        vad_threshold=0.3,
        sensitivity_threshold=0.3,
        min_trigger_interval=1.0,
    ):
        self.model_name = model_name
        self.model_folder = model_folder
        self.model_path = os.path.abspath(os.path.join(model_folder, onnx_file))
        self.vad_threshold = vad_threshold
        self.sensitivity_threshold = sensitivity_threshold
        self.min_trigger_interval = min_trigger_interval
        self.last_trigger_time = 0

        self._prepare_model()

        self.model = Model(
            wakeword_models=[self.model_path],
            inference_framework="onnx",
            vad_threshold=self.vad_threshold,
        )

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

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")

        # Convert input data to int16 format
        if indata.dtype != np.float32:
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
        self.model.cleanup()
        print("Model released.")

if __name__ == "__main__":
    detector = WakeWordDetector()
    detector.start_listening()
