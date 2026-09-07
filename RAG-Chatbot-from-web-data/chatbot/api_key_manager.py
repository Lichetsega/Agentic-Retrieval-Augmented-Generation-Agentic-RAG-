"""
Google API key rotation manager.

This module loads multiple Gemini keys from environment variables and
automatically rotates/fails over keys when one is rate-limited or fails.
"""

import os
from typing import Optional
import time

class APIKeyManager:
    """
    Manages multiple API keys with automatic failover.
    This ensures the chatbot stays 'Online' even if one key hits its usage limit.
    """

    def __init__(self):
        self.keys = []               # List to store all valid API keys
        self.key_status = {}         # Tracks if a key is active (True) or in cooldown (False)
        self.last_failure_time = {}  # Records when a key failed to calculate cooldown
        self.current_key_index = 0   # Pointer for the Round-Robin rotation

        # STEP 1: Bulk Load Keys
        # Loops infinitely until no more numbered keys are found
        i = 1
        while True:
            key = os.getenv(f"GOOGLE_API_KEY_{i}")
            if not key:
                break  # Stop the loop when we hit a missing number
            key = self._normalize_key(key)
            if not key:
                i += 1
                continue

            self.keys.append(key)
            self.key_status[key] = True
            self.last_failure_time[key] = 0
            i += 1  # Move to the next potential key

        print(f"✅ Loaded {len(self.keys)} API keys")
        if not self.keys:
            print("⚠️ WARNING: No API keys found! Bot might not respond.")

    def get_next_available_key(self) -> Optional[str]:
        """
        Implements a 'Round-Robin' selection.
        It cycles through keys one by one. If a key is 'broken' (in cooldown),
        it skips to the next one automatically.
        """
        if not self.keys:
            return None

        # Try to find a key that is currently marked as working
        for attempt in range(len(self.keys)):
            idx = (self.current_key_index + attempt) % len(self.keys)
            key = self.keys[idx]

            if self.key_status.get(key, False):
                self.current_key_index = idx
                return key

        # FAILSAFE: If all keys are in cooldown, find the one that failed longest ago
        # If it has been cooling down for more than 60 seconds, try it again.
        oldest_key = min(self.keys, key=lambda k: self.last_failure_time.get(k, 0))
        if time.time() - self.last_failure_time.get(oldest_key, 0) > 60:
            self.key_status[oldest_key] = True
            return oldest_key

        return None

    def mark_key_failed(self, key: str):
        """
        If the LLM returns a 'Quota Exceeded' error, we call this function.
        It puts the key in the 'Penalty Box' for 60 seconds.
        """
        if key in self.key_status:
            self.key_status[key] = False
            self.last_failure_time[key] = time.time()
            print(f"⚠️ Key {self._mask_key(key)} failed - cooling down 60s")

    def mark_key_success(self, key: str):
        """Resets a key to healthy status after a successful API call."""
        if key in self.key_status:
            self.key_status[key] = True
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
            print(f"⚠️ Key {self._mask_key(key)} used successfully.")

    def _mask_key(self, key: str) -> str:
        """
        Security feature: Hides most of the API key in logs so you don't
        accidentally leak your secrets in screenshots or console logs.
        """
        return f"...{key[-4:]}" if len(key) > 8 else "****"

    def _normalize_key(self, key: str) -> str:
        """
        Normalizes env-loaded API keys to avoid invalid metadata errors caused by
        accidental quotes/whitespace/newlines in .env values.
        """
        cleaned = key.strip().strip("'\"")
        # Remove invisible CR/LF characters that can break HTTP metadata.
        cleaned = cleaned.replace("\r", "").replace("\n", "")
        return cleaned

    def get_status(self) -> str:
        """Returns a visual status bar of all keys (e.g., ✅ Key1 | ⏳ Key2)."""
        statuses = []
        for i, key in enumerate(self.keys):
            status = "✅" if self.key_status.get(key, False) else "⏳"
            statuses.append(f"{status} Key{i + 1}:{self._mask_key(key)}")
        return " | ".join(statuses)

# Create a single global instance to be used across the whole app
api_key_manager = APIKeyManager()