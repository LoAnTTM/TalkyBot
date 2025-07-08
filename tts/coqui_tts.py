import os
from TTS.utils.manage import ModelManager
from TTS.api import TTS

# Cấu hình tên model bạn muốn dùng
TTS_MODEL_NAME = "tts_models/en/ljspeech/fastspeech2"
VOCODER_MODEL_NAME = "vocoder_models/en/ljspeech/hifigan"

# Thư mục lưu model
MODEL_DIR = "models"

# Tạo thư mục nếu chưa có
os.makedirs(MODEL_DIR, exist_ok=True)

# Hàm tải model nếu chưa có
def download_model_if_missing(model_name, target_dir):
    manager = ModelManager()
    model_path = os.path.join(target_dir, model_name.replace("/", "__"))
    if not os.path.exists(model_path):
        print(f"⬇️  Tải model: {model_name}")
        manager.download_model(model_name, target_dir)
    else:
        print(f"✅ Model đã có: {model_name}")
    return model_path

# Tải model nếu cần
tts_path = download_model_if_missing(TTS_MODEL_NAME, MODEL_DIR)
vocoder_path = download_model_if_missing(VOCODER_MODEL_NAME, MODEL_DIR)

# Load model từ folder
tts = TTS(
    model_path=os.path.join(tts_path, "model_file.pth"),
    config_path=os.path.join(tts_path, "config.json"),
    vocoder_path=os.path.join(vocoder_path, "model_file.pth"),
    vocoder_config_path=os.path.join(vocoder_path, "config.json"),
)



