import threading
import time
import logging
import traceback
from enum import Enum
from typing import Optional, Callable

from components.logger import get_logger

class SystemState(Enum):
    STANDBY = "STANDBY"
    LISTENING = "LISTENING"
    SPEAKING = "SPEAKING"

class StateManager:
    def __init__(self, timeout_seconds: int = 15):
        self._current_state = SystemState.STANDBY
        self.timeout_seconds = timeout_seconds
        self._state_lock = threading.RLock()
        self._last_activity_time = time.time()

        self.wake_up_event = threading.Event()
        self.stop_tts_event = threading.Event()
        self.conversation_active = threading.Event()

        self._state_callbacks = []

        self.logger = get_logger('StateManager')

    @property
    def current_state(self) -> SystemState:
        with self._state_lock:
            return self._current_state

    def is_active(self) -> bool:
        return self.current_state != SystemState.STANDBY

    def is_speaking(self) -> bool:
        return self.current_state == SystemState.SPEAKING

    def transition_to(self, new_state: SystemState, reason: str = "") -> bool:
        with self._state_lock:
            old_state = self._current_state

            if not self._is_valid_transition(old_state, new_state):
                self.logger.warning(f"Invalid transition from {old_state.name} to {new_state.name}")
                return False

            self._current_state = new_state
            self._update_activity_time()
            self._handle_state_entry(new_state)

            log_msg = f"State: {old_state.name} → {new_state.name}"
            if reason:
                log_msg += f" ({reason})"
            self.logger.info(log_msg)

            self._notify_state_callbacks(old_state, new_state, reason)
            return True

    def wake_up(self, reason: str = "Wake word detected") -> bool:
        if self.transition_to(SystemState.LISTENING, reason):
            self.wake_up_event.set()
            self.conversation_active.set()
            self.stop_tts_event.clear()
            return True
        return False

    def go_to_standby(self, reason: str = "Timeout or sleep command") -> bool:
        if self.transition_to(SystemState.STANDBY, reason):
            self._enter_standby()
            return True
        return False

    def start_speaking(self, reason: str = "Response ready") -> bool:
        return self.transition_to(SystemState.SPEAKING, reason)

    def interrupt_speaking(self, reason: str = "User interrupted") -> bool:
        if self.current_state == SystemState.SPEAKING:
            self.stop_tts_event.set()
            return self.transition_to(SystemState.LISTENING, reason)
        return False

    def update_activity(self):
        with self._state_lock:
            self._update_activity_time()
            self.logger.debug(f"🔄 Activity updated at {time.time():.2f}")

    def check_timeout(self) -> bool:
        if not self.is_active():
            return False
        if time.time() - self._last_activity_time > self.timeout_seconds:
            self.go_to_standby("Timeout")
            return True
        return False

    def add_state_callback(self, callback: Callable):
        self._state_callbacks.append(callback)

    def get_state_info(self) -> dict:
        with self._state_lock:
            elapsed = time.time() - self._last_activity_time
            return {
                'current_state': self.current_state.name,
                'is_active': self.is_active(),
                'is_speaking': self.is_speaking(),
                'time_since_activity': round(elapsed, 2),
                'timeout_seconds': self.timeout_seconds,
                'conversation_active': self.conversation_active.is_set(),
                'wake_up_event': self.wake_up_event.is_set(),
                'stop_tts_event': self.stop_tts_event.is_set()
            }

    # ──────────────────────────────
    # Internal helpers
    # ──────────────────────────────

    def _is_valid_transition(self, from_state: SystemState, to_state: SystemState) -> bool:
        valid_transitions = {
            SystemState.STANDBY: [SystemState.LISTENING],
            SystemState.LISTENING: [SystemState.SPEAKING],
            SystemState.SPEAKING: [SystemState.LISTENING, SystemState.STANDBY]
        }
        return to_state in valid_transitions.get(from_state, [])

    def _handle_state_entry(self, state: SystemState):
        if state == SystemState.STANDBY:
            self._enter_standby()
        elif state == SystemState.LISTENING:
            self.stop_tts_event.clear()
        elif state == SystemState.SPEAKING:
            self.stop_tts_event.clear()

    def _enter_standby(self):
        self.conversation_active.clear()
        self.wake_up_event.clear()
        self.stop_tts_event.set()

    def _update_activity_time(self):
        self._last_activity_time = time.time()

    def _notify_state_callbacks(self, old_state: SystemState, new_state: SystemState, reason: str):
        for callback in self._state_callbacks:
            try:
                callback(old_state, new_state, reason)
            except Exception:
                self.logger.error("Error in state callback:\n" + traceback.format_exc())
