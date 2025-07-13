import numpy as np
import sounddevice as sd
from TTS.api import TTS

def test_model(model_name, text):
    # print(f"\n🔊 Testing model: {model_name}")
    # Load model
    tts = TTS(model_name=model_name)

    # Generate waveform
    wav = tts.tts(text)

    # Normalize to -1.0 ~ 1.0
    wav = wav / np.abs(wav).max()

    # Play
    sd.play(wav, samplerate=tts.synthesizer.output_sample_rate)
    sd.wait()

if __name__ == "__main__":
    # Use only VITS LJSpeech model
    model = "tts_models/en/ljspeech/vits"
    
    # print("Text: ")
    test_text = "Two antennas met on a roof, fell in love, and got married. The ceremony wasn't much, but the reception was excellent."
                
#     test_text = """Follow two young explorers on a monorhyme (all the lines end in the same rhyme) adventure that takes you across the sea to the mysterious Island of Bum Bum Ba Loo. You'll meet the King and Queen, dance with Bum Bum Balites, and learn the secret to Bum Berry Goo! The only problem is finding your way back again...
# The Island of Bum Bum Ba Loo is a bedtime tale about discovery, with an ending to encourage the explorer in us all!

# Have you sailed to the island of Bum Bum Ba Loo?

# It’s something that all great explorers must do"""
    test_model(model, test_text)
