import logging
import os
import sys
from typing import Optional

class TalkyBotLogger:
    _instance: Optional['TalkyBotLogger'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.log_file = "TalkyBot.log"
        self._setup()
        self._initialized = True

    def _setup(self):
        logger = logging.getLogger("TalkyBot")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()

        # Format
        fmt_file = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fmt_console = logging.Formatter('%(asctime)s | %(levelname)-5s | %(name)-12s | %(message)s',
                                        datefmt='%H:%M:%S')

        # File handler
        try:
            fh = logging.FileHandler(self.log_file, encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt_file)
            logger.addHandler(fh)
        except Exception as e:
            print(f"⚠️ Cannot create file handler: {e}")

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt_console)
        logger.addHandler(ch)

        logger.info("=" * 60)
        logger.info("TalkyBot Logging System Initialized")
        logger.info(f"Log file: {os.path.abspath(self.log_file)}")
        logger.info("=" * 60)

        self.logger = logger

    def get(self, name: Optional[str] = None) -> logging.Logger:
        return logging.getLogger(f"TalkyBot.{name}") if name else self.logger

    def close(self):
        self.logger.info("Shutting down logging system")
        for h in self.logger.handlers:
            h.close()
        self.logger.handlers.clear()


# --- Global Accessors ---

_logger_instance: Optional[TalkyBotLogger] = None

def get_logger(name: Optional[str] = None) -> logging.Logger:
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TalkyBotLogger()
    return _logger_instance.get(name)

def setup_logging():
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TalkyBotLogger()

def close_logging():
    global _logger_instance
    if _logger_instance:
        _logger_instance.close()
        _logger_instance = None
