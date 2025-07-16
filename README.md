# TalkyBot

A conversational robot with voice recognition, wake word detection, and text-to-speech capabilities.

## Prerequisites

### Install Miniconda (if not already installed)

1. **Download Miniconda:**

   - Visit: https://docs.conda.io/en/latest/miniconda.html
   - Choose the appropriate installer for your system

2. **Install Miniconda:**

   ```bash
   # For macOS/Linux
   bash Miniconda3-latest-MacOSX-arm64.sh

   # Follow the installation prompts
   # Restart your terminal after installation
   ```

3. **Verify installation:**
   ```bash
   conda --version
   ```

## Setup

### Option 1: Automatic Setup (Recommended)

```bash
# Make the script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

The script will automatically:

- Create conda environment named "TalkyBot" with Python 3.10
- Install all dependencies from requirements.txt
- Install additional packages manually
- Create necessary directories

### Option 2: Manual Setup

### 1. Create Conda Environment

```bash
conda create -n talkybot python=3.10
conda activate talkybot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Manual Installations

```bash
#TensorFlow
pip install tensorflow
# TensorFlow for macOS (Apple Silicon)
pip install tensorflow-macos

# OpenWakeWord
pip install openwakeword==0.6.0

# TTS (Text-to-Speech)
pip install TTS

# ONNX Runtime
pip install onnxruntime
```

## Current Features (July 14, 2025)

✅ **VAD (Voice Activity Detection)** - Detects voice activity  
✅ **Wake Word Detection** - Detects wake words work for only "Alexa"  
✅ **TTS (Text-to-Speech)** - Converts text to speech

## Models Used in This Project

### Wake Word Detection

- **OpenWakeWord Models**: https://github.com/dscripka/openWakeWord
  - `alexa_v0.1.onnx` - Pre-trained "Alexa" wake word detection model
  - `embedding_model.onnx` - Audio embedding extraction
  - `melspectrogram.onnx` - Mel-spectrogram feature extraction
  - `silero_vad.onnx` - Voice Activity Detection

### Voice Activity Detection (VAD)

- **Silero VAD Model**: https://github.com/snakers4/silero-vad
  - Pre-trained VAD model for speech/non-speech classification
  - Optimized for real-time processing

### Text-to-Speech (TTS)

- **Coqui TTS Models**: https://github.com/coqui-ai/TTS
  - Various pre-trained TTS models available
  - Support for multiple languages and voices
  - Models downloaded automatically based on configuration

## Project Structure

```
TalkyBot/
├── main.py                 # Main application entry point
├── setup.sh                # Automated setup script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── audio/
│   ├── mic_stream.py       # Microphone stream handling
│   ├── recorder.py         # Audio recording functionality
│   └── speaker.py          # Audio output handling
├── components/
│   ├── brain.py            # Conversational AI using BlenderBot
│   ├── stt.py              # Speech-to-text using Vosk
│   ├── tts.py              # Text-to-speech
│   ├── vad.py              # Voice activity detection
│   └── wakeword.py         # Wake word detection using OpenWakeWord
└── models/                 # Downloaded models (auto-created)
    ├── openwakeword/       # OpenWakeWord models
    │   ├── alexa_v0.1.onnx
    │   ├── alexa_v0.1.tflite
    │   ├── embedding_model.onnx
    │   └── embedding_model.tflite
    └── vosk/               # Vosk speech recognition models
        └── vosk-model-small-en-us-0.15/
```

## Usage

### Test Wake Word Detection

```bash
cd components
python wakeword.py
```

Say "Alexa" to test wake word detection.

### Test Voice Activity Detection

```bash
cd components
python vad.py
```

### Test Text-to-Speech

```bash
cd components
python tts.py
```

## Notes

- Microphone access required for testing
- Some models will be downloaded automatically on first run
- Make sure to activate the conda environment before running scripts
