"""
core/audio_engine.py
====================
Simple audio engine for the Yautja Bio-Mask using pygame.
"""

import os
import threading
import pygame

class AudioEngine:
    """Handles loading and playing HUD sound effects."""

    def __init__(self, sound_dir: str = "sounds") -> None:
        self.sound_dir = sound_dir
        self.enabled = False
        try:
            pygame.mixer.init()
            self.enabled = True
        except Exception as e:
            print(f"[warn] Audio Engine failed to initialize: {e}")

        self.sounds: dict[str, pygame.mixer.Sound] = {}
        if self.enabled:
            self._load_sounds()

    def _load_sounds(self) -> None:
        """Load all .wav and .mp3 files from the sounds directory."""
        if not os.path.exists(self.sound_dir):
            return

        for file in os.listdir(self.sound_dir):
            if file.endswith((".wav", ".mp3")):
                name = os.path.splitext(file)[0].upper()
                try:
                    self.sounds[name] = pygame.mixer.Sound(os.path.join(self.sound_dir, file))
                    print(f"  [Audio] Loaded: {name}")
                except Exception as e:
                    print(f"  [warn] Failed to load sound {file}: {e}")

    def play(self, sound_name: str) -> None:
        """Play a loaded sound by name (case-insensitive)."""
        if not self.enabled:
            return

        name = sound_name.upper()
        if name in self.sounds:
            # Play in a background thread to avoid any micro-stutters
            threading.Thread(target=self.sounds[name].play, daemon=True).start()

# Global instance
engine = AudioEngine()
