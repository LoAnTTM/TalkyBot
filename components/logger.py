import logging
import os
import sys
from datetime import datetime
from typing import Optional


class TalkyBotLogger:
    """
    Centralized logging system for TalkyBot.
    Provides structured logging to both console and file with proper formatting.
    """
    
    _instance: Optional['TalkyBotLogger'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.log_file = "TalkyBot.log"
        self.setup_logging()
        self._initialized = True
    
    def setup_logging(self):
        """Setup logging configuration for both file and console output"""
        
        # Create root logger
        self.logger = logging.getLogger('TalkyBot')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-5s | %(name)-12s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler - logs everything to TalkyBot.log
        try:
            file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file handler: {e}")
        
        # Console handler - logs INFO and above to console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Back to INFO for production
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Log startup
        self.logger.info("=" * 60)
        self.logger.info("TalkyBot Logging System Initialized")
        self.logger.info(f"Log file: {os.path.abspath(self.log_file)}")
        self.logger.info("=" * 60)
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """Get a logger instance for a specific component"""
        if name:
            return logging.getLogger(f'TalkyBot.{name}')
        return self.logger
    
    def log_system_event(self, event: str, details: str = "", level: int = logging.INFO):
        """Log system-level events"""
        message = f"SYSTEM: {event}"
        if details:
            message += f" - {details}"
        self.logger.log(level, message)
    
    def log_state_change(self, old_state: str, new_state: str, reason: str = ""):
        """Log state transitions"""
        message = f"STATE: {old_state} → {new_state}"
        if reason:
            message += f" ({reason})"
        self.logger.info(message)
    
    def log_audio_event(self, event: str, details: str = ""):
        """Log audio-related events"""
        message = f"AUDIO: {event}"
        if details:
            message += f" - {details}"
        self.logger.debug(message)
    
    def log_stt_event(self, event: str, text: str = "", confidence: float = None):
        """Log speech-to-text events"""
        message = f"STT: {event}"
        if text:
            message += f" - '{text}'"
        if confidence is not None:
            message += f" (confidence: {confidence:.2f})"
        self.logger.info(message)
    
    def log_tts_event(self, event: str, text: str = ""):
        """Log text-to-speech events"""
        message = f"TTS: {event}"
        if text:
            message += f" - '{text}'"
        self.logger.info(message)
    
    def log_wake_word_event(self, event: str, word: str = "", score: float = None):
        """Log wake word detection events"""
        message = f"WAKE: {event}"
        if word:
            message += f" - '{word}'"
        if score is not None:
            message += f" (score: {score:.3f})"
        self.logger.info(message)
    
    def log_vad_event(self, event: str, duration: float = None):
        """Log voice activity detection events"""
        message = f"VAD: {event}"
        if duration is not None:
            message += f" - {duration:.1f}s"
        self.logger.debug(message)
    
    def log_chatbot_event(self, event: str, input_text: str = "", response: str = ""):
        """Log chatbot interactions"""
        message = f"CHAT: {event}"
        if input_text:
            message += f" - Input: '{input_text}'"
        if response:
            message += f" - Response: '{response}'"
        self.logger.info(message)
    
    def log_error(self, component: str, error: str, exception: Exception = None):
        """Log errors with component context"""
        message = f"ERROR in {component}: {error}"
        if exception:
            self.logger.error(message, exc_info=True)
        else:
            self.logger.error(message)
    
    def log_performance(self, component: str, operation: str, duration: float):
        """Log performance metrics"""
        message = f"PERF: {component}.{operation} took {duration:.3f}s"
        self.logger.debug(message)
    
    def close(self):
        """Close logging handlers"""
        self.log_system_event("Shutting down logging system")
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()


# Global logger instance
_talky_logger = None

def get_logger(name: str = None) -> logging.Logger:
    """Get the global TalkyBot logger instance"""
    global _talky_logger
    if _talky_logger is None:
        _talky_logger = TalkyBotLogger()
    return _talky_logger.get_logger(name)

def setup_logging():
    """Initialize the global logging system"""
    global _talky_logger
    if _talky_logger is None:
        _talky_logger = TalkyBotLogger()
    return _talky_logger

def log_system_event(event: str, details: str = "", level: int = logging.INFO):
    """Convenience function for logging system events"""
    global _talky_logger
    if _talky_logger is None:
        _talky_logger = TalkyBotLogger()
    _talky_logger.log_system_event(event, details, level)

def log_state_change(old_state: str, new_state: str, reason: str = ""):
    """Convenience function for logging state changes"""
    global _talky_logger
    if _talky_logger is None:
        _talky_logger = TalkyBotLogger()
    _talky_logger.log_state_change(old_state, new_state, reason)

def log_error(component: str, error: str, exception: Exception = None):
    """Convenience function for logging errors"""
    global _talky_logger
    if _talky_logger is None:
        _talky_logger = TalkyBotLogger()
    _talky_logger.log_error(component, error, exception)

def close_logging():
    """Close the logging system"""
    global _talky_logger
    if _talky_logger is not None:
        _talky_logger.close()
        _talky_logger = None


# Test the logging system
if __name__ == "__main__":
    import time
    
    print("🔧 Testing TalkyBot Logging System...")
    
    # Initialize logging
    logger_system = setup_logging()
    
    # Test different types of logs
    log_system_event("Testing system event logging")
    log_state_change("STANDBY", "LISTENING", "Wake word detected")
    
    # Get component loggers
    stt_logger = get_logger("STT")
    tts_logger = get_logger("TTS")
    vad_logger = get_logger("VAD")
    
    # Test component logs
    stt_logger.info("Testing STT logger")
    tts_logger.info("Testing TTS logger")
    vad_logger.debug("Testing VAD logger (debug level)")
    
    # Test structured logging methods
    logger_system.log_wake_word_event("Wake word detected", "alexa", 0.85)
    logger_system.log_stt_event("Speech recognized", "hello world", 0.95)
    logger_system.log_tts_event("Speaking response", "Hello! How can I help you?")
    logger_system.log_vad_event("Speech detected", 2.5)
    logger_system.log_chatbot_event("Response generated", "hello", "Hi there!")
    
    # Test error logging
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        log_error("TestComponent", "This is a test error", e)
    
    # Test performance logging
    start_time = time.time()
    time.sleep(0.1)  # Simulate some work
    duration = time.time() - start_time
    logger_system.log_performance("TestComponent", "test_operation", duration)
    
    print("\n✅ Logging test completed! Check TalkyBot.log for file output.")
    print(f"📝 Log file location: {os.path.abspath('TalkyBot.log')}")
    
    # Close logging
    close_logging() 