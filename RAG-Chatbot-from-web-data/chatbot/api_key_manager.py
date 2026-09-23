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
        self.failure_counts = {}     # Total failure attempts per key
        self.quota_failure_counts = {} # Quota/429 failures per key
        self.consecutive_failures = {} # Consecutive quota failures per key
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
            self.failure_counts[key] = 0
            self.quota_failure_counts[key] = 0
            self.consecutive_failures[key] = 0
            i += 1  # Move to the next potential key

        try:
            print(f"✅ Loaded {len(self.keys)} API keys")
        except UnicodeEncodeError:
            print(f"[OK] Loaded {len(self.keys)} API keys")
        if not self.keys:
            try:
                print("⚠️ WARNING: No API keys found! Bot might not respond.")
            except UnicodeEncodeError:
                print("WARNING: No API keys found! Bot might not respond.")

    def is_client_error(self, err: Exception) -> bool:
        """
        Determines if an error is due to bad user input, invalid argument,
        prompt safety blocks, or payload format errors.
        Client errors should NOT trigger key rotation or penalize healthy API keys.
        """
        if err is None:
            return False
        msg = str(err).lower()
        client_markers = [
            "invalid_argument",
            "invalid argument",
            "bad_request",
            "bad request",
            "400",
            "context_length_exceeded",
            "token_limit",
            "payload too large",
            "valueerror",
        ]
        return any(marker in msg for marker in client_markers)

    def is_quota_error(self, err: Exception) -> bool:
        """
        Determines if an error is a true Google API 429 / Rate Limit / Quota Exceeded error.
        Only these errors should trigger key rotation and mark the key for cooldown.
        """
        if err is None:
            return True  # Default fallback if unspecified
        msg = str(err).lower()
        quota_markers = [
            "quota",
            "429",
            "resource_exhausted",
            "rate_limit",
            "rate limit",
            "daily_limit",
            "api_key_invalid",
            "unauthenticated",
            "permission_denied",
        ]
        return any(marker in msg for marker in quota_markers)

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
        cooldown_target = 300 if self.consecutive_failures.get(oldest_key, 0) >= 3 else 60
        if time.time() - self.last_failure_time.get(oldest_key, 0) > cooldown_target:
            self.key_status[oldest_key] = True
            return oldest_key

        return None

    def mark_key_failed(self, key: str, error: Optional[Exception] = None):
        """
        If an API call fails:
        - If it's a Client Error (400 / Bad Request / Safety), DO NOT mark key as failed or rotate!
        - If it's a Quota / Rate Limit Error (429), put key in cooldown and track failure counters.
        """
        if key not in self.key_status:
            return

        # Safeguard: Do NOT penalize API keys for client-side bad inputs or safety blocks!
        if error is not None and self.is_client_error(error):
            print(f"🛑 Client input error on key {self._mask_key(key)}: '{error}' - Key remains healthy, rotation skipped.")
            return

        self.failure_counts[key] = self.failure_counts.get(key, 0) + 1
        self.quota_failure_counts[key] = self.quota_failure_counts.get(key, 0) + 1
        self.consecutive_failures[key] = self.consecutive_failures.get(key, 0) + 1
        self.key_status[key] = False
        self.last_failure_time[key] = time.time()

        consecutive = self.consecutive_failures[key]
        cooldown_sec = 300 if consecutive >= 3 else 60
        print(
            f"⚠️ Key {self._mask_key(key)} quota failed (total fails: {self.quota_failure_counts[key]}, "
            f"consecutive: {consecutive}) - cooling down for {cooldown_sec}s"
        )

    def mark_key_success(self, key: str):
        """Resets a key to healthy status after a successful API call."""
        if key in self.key_status:
            self.key_status[key] = True
            self.consecutive_failures[key] = 0
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)

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
        Accepts both standard 'AIza...' and newer 'AQ....' Google Gemini API keys.
        """
        cleaned = key.strip().strip("'\"")
        # Remove invisible CR/LF characters that can break HTTP metadata.
        cleaned = cleaned.replace("\r", "").replace("\n", "")
        return cleaned

    def get_status(self) -> str:
        """Returns a visual status bar of all keys (e.g., ✅ Key1 (0 fails) | ⏳ Key2 (2 fails))."""
        statuses = []
        for i, key in enumerate(self.keys):
            status = "✅" if self.key_status.get(key, False) else "⏳"
            fails = self.quota_failure_counts.get(key, 0)
            statuses.append(f"{status} Key{i + 1}:{self._mask_key(key)} ({fails} fails)")
        return " | ".join(statuses)

    def get_stats(self) -> dict:
        """Returns detailed health dictionary of all managed API keys."""
        return {
            "total_keys": len(self.keys),
            "active_keys": sum(1 for k in self.keys if self.key_status.get(k, False)),
            "per_key": [
                {
                    "masked_key": self._mask_key(k),
                    "active": self.key_status.get(k, False),
                    "total_failures": self.failure_counts.get(k, 0),
                    "quota_failures": self.quota_failure_counts.get(k, 0),
                    "consecutive_failures": self.consecutive_failures.get(k, 0),
                }
                for k in self.keys
            ]
        }

# Create a single global instance to be used across the whole app
api_key_manager = APIKeyManager()