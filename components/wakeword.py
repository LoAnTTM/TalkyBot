import os
import time
import numpy as np
import sounddevice as sd
import openwakeword
from openwakeword.model import Model

# === CẤU HÌNH ===
model_folder = "models/openwakeword"
model_name = "alexa"
onnx_file = "alexa_v0.1.onnx"
model_path = os.path.abspath(os.path.join(model_folder, onnx_file))

vad_threshold = 0.3               # độ nhạy VAD tích hợp
sensitivity_threshold = 0.3       # ngưỡng phát hiện wake word
min_trigger_interval = 1.0        # giây giữa 1 lần phát hiện liên tiếp

last_trigger_time = 0  # dùng để tránh phát hiện liên tục

# === TẠO THƯ MỤC VÀ TẢI MODEL ===
os.makedirs(model_folder, exist_ok=True)

if not os.path.isfile(model_path):
    print(f"➡️ Đang tải model '{model_name}' vào {model_folder}...")
    openwakeword.utils.download_models(
        model_names=[model_name],
        target_directory=model_folder
    )
else:
    print(f"✅ Model đã tồn tại tại: {model_path}")

# === KHỞI TẠO MODEL WAKEWORD ===
model = Model(
    wakeword_models=[model_path],
    inference_framework="onnx",
    vad_threshold=vad_threshold
)

# === HÀM CALLBACK XỬ LÝ ÂM THANH ===
def callback(indata, frames, time_info, status):
    global last_trigger_time

    if status:
        print(f"⚠️ Audio status: {status}")

    audio_int16 = (indata[:, 0] * 32767).astype(np.int16)

    try:
        scores = model.predict(audio_int16)
        current_time = time.time()
        for wake_word, score in scores.items():
            if score > sensitivity_threshold and (current_time - last_trigger_time > min_trigger_interval):
                print(f"🔊 Wake word '{wake_word}' detected! (score: {score:.3f})")
                last_trigger_time = current_time
    except Exception as e:
        print(f"❌ Predict error: {e}")

# === GHI ÂM MICROPHONE ===
print("🎤 Đang nghe... (nói 'Alexa' để test)")
try:
    with sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype="float32",
        blocksize=512,
        callback=callback,
    ):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\n🛑 Đã dừng.")
finally:
    model.cleanup()
    print("🧹 Đã giải phóng model.")
