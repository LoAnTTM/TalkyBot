"""
Audio module for TalkyBot
Contains audio input/output components: microphone streaming, audio recording, and speaker output.
"""

from .mic_stream import AudioStream
from .recorder import SpeechRecorder
from .speaker import Speaker

__all__ = [
    'AudioStream',
    'SpeechRecorder', 
    'Speaker'
]
