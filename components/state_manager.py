import threading
import time
import logging
from enum import Enum
from typing import Optional, Callable


class SystemState(Enum):
    """System states for TalkyBot"""
    STANDBY = "STANDBY"     # Wake word detection only, system waiting for activation
    LISTENING = "LISTENING" # Active, waiting for user speech via VAD
    PROCESSING = "PROCESSING" # STT + Brain processing user input
    SPEAKING = "SPEAKING"   # TTS playing response


class StateManager:
    """
    Centralized state manager for TalkyBot system.
    Manages state transitions between STANDBY and ACTIVE modes with thread-safe operations.
    """
    
    def __init__(self, timeout_seconds: int = 15):
        self.current_state = SystemState.STANDBY
        self.timeout_seconds = timeout_seconds
        self._state_lock = threading.RLock()
        self._last_activity_time = time.time()
        
        # Events for thread communication
        self.wake_up_event = threading.Event()  # Wake word detected
        self.stop_tts_event = threading.Event()  # Stop TTS immediately
        self.conversation_active = threading.Event()  # Conversation in progress
        
        # State change callbacks
        self._state_callbacks = []
        
        # Sleep keywords that trigger STANDBY
        self.sleep_keywords = [
            'bye bye', 'goodbye', 'go to sleep', 'sleep', 'stop listening', 
            'see you later', 'goodnight', 'shut down', 'power off', 
            'deactivate', 'turn off'
        ]
        
        # Logging
        self.logger = logging.getLogger('StateManager')
        
    def get_current_state(self) -> SystemState:
        """Get current system state (thread-safe)"""
        with self._state_lock:
            return self.current_state
    
    def is_active(self) -> bool:
        """Check if system is in any active state (not STANDBY)"""
        with self._state_lock:
            return self.current_state != SystemState.STANDBY
    
    def is_speaking(self) -> bool:
        """Check if system is currently speaking"""
        with self._state_lock:
            return self.current_state == SystemState.SPEAKING
    
    def transition_to(self, new_state: SystemState, reason: str = "") -> bool:
        """
        Transition to new state with logging and callbacks
        Returns True if transition was successful
        """
        with self._state_lock:
            old_state = self.current_state
            
            # Validate state transition
            if not self._is_valid_transition(old_state, new_state):
                self.logger.warning(f"Invalid transition from {old_state.value} to {new_state.value}")
                return False
            
            # Execute transition
            self.current_state = new_state
            self._update_activity_time()
            
            # Handle state-specific actions
            self._handle_state_entry(new_state, old_state)
            
            # Log transition
            log_msg = f"State: {old_state.value} → {new_state.value}"
            if reason:
                log_msg += f" ({reason})"
            self.logger.info(log_msg)
            
            # Notify callbacks
            self._notify_state_callbacks(old_state, new_state, reason)
            
            return True
    
    def wake_up(self, reason: str = "Wake word detected"):
        """Wake up from STANDBY to LISTENING"""
        if self.transition_to(SystemState.LISTENING, reason):
            self.wake_up_event.set()
            self.conversation_active.set()
            self.stop_tts_event.clear()
    
    def go_to_standby(self, reason: str = "Timeout or sleep command"):
        """Return to STANDBY state"""
        if self.transition_to(SystemState.STANDBY, reason):
            self.conversation_active.clear()
            self.wake_up_event.clear()
            self.stop_tts_event.set()  # Stop any ongoing TTS
    
    def start_processing(self, reason: str = "Speech detected"):
        """Start processing user input"""
        self.transition_to(SystemState.PROCESSING, reason)
    
    def start_speaking(self, reason: str = "Response ready"):
        """Start TTS speaking"""
        self.transition_to(SystemState.SPEAKING, reason)
    
    def interrupt_speaking(self, reason: str = "User interrupted"):
        """Interrupt TTS and return to listening"""
        if self.current_state == SystemState.SPEAKING:
            self.stop_tts_event.set()
            self.transition_to(SystemState.LISTENING, reason)
    
    def update_activity(self):
        """Update last activity time (call when user speech detected)"""
        with self._state_lock:
            self._update_activity_time()
    
    def check_timeout(self) -> bool:
        """
        Check if conversation has timed out (15 seconds of inactivity)
        Returns True if timeout occurred and state changed to STANDBY
        """
        with self._state_lock:
            if self.current_state == SystemState.STANDBY:
                return False
                
            elapsed = time.time() - self._last_activity_time
            if elapsed > self.timeout_seconds:
                self.go_to_standby(f"Timeout after {elapsed:.1f}s")
                return True
            return False
    
    def check_sleep_keywords(self, text: str) -> bool:
        """
        Check if text contains sleep keywords
        Returns True if sleep keyword detected and state changed to STANDBY
        """
        text_lower = text.lower().strip()
        for keyword in self.sleep_keywords:
            if keyword in text_lower:
                self.go_to_standby(f"Sleep keyword detected: '{keyword}'")
                return True
        return False
    
    def add_state_callback(self, callback: Callable):
        """Add callback function for state changes"""
        self._state_callbacks.append(callback)
    
    def _is_valid_transition(self, from_state: SystemState, to_state: SystemState) -> bool:
        """Validate if state transition is allowed"""
        # Define valid transitions
        valid_transitions = {
            SystemState.STANDBY: [SystemState.LISTENING],
            SystemState.LISTENING: [SystemState.PROCESSING, SystemState.SPEAKING, SystemState.STANDBY],
            SystemState.PROCESSING: [SystemState.SPEAKING, SystemState.LISTENING, SystemState.STANDBY],
            SystemState.SPEAKING: [SystemState.LISTENING, SystemState.STANDBY]
        }
        
        return to_state in valid_transitions.get(from_state, [])
    
    def _handle_state_entry(self, new_state: SystemState, old_state: SystemState):
        """Handle actions when entering a new state"""
        if new_state == SystemState.STANDBY:
            # Clear all active events when going to standby
            self.conversation_active.clear()
            self.stop_tts_event.set()
        elif new_state == SystemState.LISTENING:
            # Clear TTS stop event when starting to listen
            self.stop_tts_event.clear()
        elif new_state == SystemState.SPEAKING:
            # Clear stop event when starting to speak
            self.stop_tts_event.clear()
    
    def _update_activity_time(self):
        """Update last activity timestamp"""
        self._last_activity_time = time.time()
    
    def _notify_state_callbacks(self, old_state: SystemState, new_state: SystemState, reason: str):
        """Notify all registered callbacks about state change"""
        for callback in self._state_callbacks:
            try:
                callback(old_state, new_state, reason)
            except Exception as e:
                self.logger.error(f"Error in state callback: {e}")
    
    def get_state_info(self) -> dict:
        """Get comprehensive state information"""
        with self._state_lock:
            elapsed = time.time() - self._last_activity_time
            return {
                'current_state': self.current_state.value,
                'is_active': self.is_active(),
                'is_speaking': self.is_speaking(),
                'time_since_activity': elapsed,
                'timeout_seconds': self.timeout_seconds,
                'conversation_active': self.conversation_active.is_set(),
                'wake_up_event': self.wake_up_event.is_set(),
                'stop_tts_event': self.stop_tts_event.is_set()
            }


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create state manager
    state_manager = StateManager(timeout_seconds=5)  # Short timeout for testing
    
    # Add callback for state changes
    def state_callback(old_state, new_state, reason):
        print(f"📊 State Change: {old_state.value} → {new_state.value} ({reason})")
    
    state_manager.add_state_callback(state_callback)
    
    print("🔧 Testing StateManager...")
    print(f"Initial state: {state_manager.get_current_state().value}")
    
    # Test wake up
    print("\n🎤 Testing wake up...")
    state_manager.wake_up("Test wake word")
    print(f"State: {state_manager.get_current_state().value}")
    
    # Test processing
    print("\n🔄 Testing processing...")
    state_manager.start_processing("Test speech detected")
    print(f"State: {state_manager.get_current_state().value}")
    
    # Test speaking
    print("\n🔊 Testing speaking...")
    state_manager.start_speaking("Test response ready")
    print(f"State: {state_manager.get_current_state().value}")
    
    # Test interrupt
    print("\n⚡ Testing interrupt...")
    state_manager.interrupt_speaking("Test user interrupt")
    print(f"State: {state_manager.get_current_state().value}")
    
    # Test sleep keywords
    print("\n😴 Testing sleep keywords...")
    result = state_manager.check_sleep_keywords("goodbye see you later")
    print(f"Sleep detected: {result}, State: {state_manager.get_current_state().value}")
    
    # Test timeout
    print("\n⏰ Testing timeout...")
    state_manager.wake_up("Test for timeout")
    print(f"State: {state_manager.get_current_state().value}")
    print("Waiting for timeout...")
    time.sleep(6)  # Wait longer than timeout
    timeout_occurred = state_manager.check_timeout()
    print(f"Timeout occurred: {timeout_occurred}, State: {state_manager.get_current_state().value}")
    
    print("\n📊 Final state info:")
    import json
    print(json.dumps(state_manager.get_state_info(), indent=2)) 