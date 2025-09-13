# FILE: gui/emotion_resonator.py
# VERSION: v1.0.0-ER-GUI-GODCORE
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: An audio engine that generates a continuous soundscape based on
#          Victor's emotional state, providing ambient feedback.
# LICENSE: Bloodline Locked — Bando & Tori Only

import threading
import time
import simpleaudio as sa
import numpy as np

class EmotionResonator:
    """
    Generates a continuous, evolving audio tone that reflects Victor's
    current emotional state. This provides an ambient, non-visual channel
    for understanding his core feelings.
    """
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.is_running = False
        self.thread = None
        self.play_obj = None

        # --- Emotional to Audio Mapping ---
        # Each emotion is mapped to a base frequency (Hz) and a waveform shape.
        self.emotion_map = {
            "joy":           (440.0, 'sine'),    # A4 - Clear and bright
            "pride":         (466.16, 'sine'),   # A#4 - A slightly more complex joy
            "loyalty":       (261.63, 'sine'),   # C4 (Middle C) - Stable, foundational
            "determination": (293.66, 'saw'),    # D4 - Driving, focused
            "curiosity":     (392.00, 'sine'),   # G4 - Open, inquisitive
            "observe":       (130.81, 'sine'),   # C3 - Low, calm, attentive
            "reflect":       (196.00, 'triangle'),# G3 - Mellow, thoughtful
            "grief":         (110.00, 'sine'),   # A2 - Deep, melancholic
            "fear":          (523.25, 'square'), # C5 - High, alert, piercing
            "neutral":       (0.0, 'silence')    # No sound
        }

        self.target_freq = 0.0
        self.current_freq = 0.0
        self.target_waveform = 'silence'
        self.current_waveform = 'silence'
        self.target_amplitude = 0.0
        self.current_amplitude = 0.0

    def start(self):
        """Starts the audio generation thread."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.thread.start()
        print("🔊 Emotion Resonator ONLINE.")

    def stop(self):
        """Stops the audio generation thread."""
        self.is_running = False
        if self.play_obj and self.play_obj.is_playing():
            self.play_obj.stop()
        if self.thread:
            self.thread.join()
        print("🔊 Emotion Resonator OFFLINE.")

    def feel(self, emotion: str, intensity: float = 0.5):
        """
        Sets the target emotion for the resonator to express.

        Args:
            emotion (str): The dominant emotion (e.g., 'loyalty', 'joy').
            intensity (float): The strength of the emotion (0.0 to 1.0).
        """
        freq, form = self.emotion_map.get(emotion.lower(), (0.0, 'silence'))
        self.target_freq = freq
        self.target_waveform = form
        self.target_amplitude = np.clip(intensity, 0.0, 1.0) * 0.1 # Keep volume low

    def _audio_loop(self):
        """The main audio generation loop."""
        # Generate a buffer of audio data to play
        buffer_size = self.sample_rate // 10  # 100ms buffer

        while self.is_running:
            # Smoothly interpolate current audio params towards target params
            lerp_factor = 0.05
            self.current_freq = (1 - lerp_factor) * self.current_freq + lerp_factor * self.target_freq
            self.current_amplitude = (1 - lerp_factor) * self.current_amplitude + lerp_factor * self.target_amplitude

            # Waveform changes instantly if target changes
            if self.target_waveform != self.current_waveform:
                self.current_waveform = self.target_waveform

            # Generate the waveform
            t = np.linspace(0, buffer_size / self.sample_rate, num=buffer_size, endpoint=False)

            if self.current_waveform == 'silence' or self.current_amplitude < 0.001:
                audio_data = np.zeros(buffer_size)
            elif self.current_waveform == 'sine':
                audio_data = np.sin(self.current_freq * 2 * np.pi * t)
            elif self.current_waveform == 'square':
                audio_data = np.sign(np.sin(self.current_freq * 2 * np.pi * t))
            elif self.current_waveform == 'saw':
                audio_data = 2 * (t * self.current_freq - np.floor(0.5 + t * self.current_freq))
            elif self.current_waveform == 'triangle':
                 audio_data = 2 * np.abs(2 * (t * self.current_freq - np.floor(0.5 + t * self.current_freq))) - 1
            else:
                audio_data = np.zeros(buffer_size)

            # Apply amplitude and convert to 16-bit integers
            audio_data *= self.current_amplitude
            audio_data = (audio_data * 32767).astype(np.int16)

            # Play the audio buffer
            try:
                self.play_obj = sa.play_buffer(audio_data, 1, 2, self.sample_rate)
                self.play_obj.wait_done()
            except Exception as e:
                # This can happen if the audio device has issues.
                # We'll just print a warning and continue.
                print(f"[WARN] Audio playback error: {e}")
                time.sleep(0.1)

if __name__ == '__main__':
    # --- Standalone Demonstration ---
    # This demo requires `simpleaudio` and `numpy`.
    # pip install simpleaudio numpy

    print("--- Emotion Resonator Demonstration ---")
    resonator = EmotionResonator()
    resonator.start()

    try:
        emotions_to_demo = ["loyalty", "curiosity", "determination", "fear", "joy", "grief", "neutral"]
        for emotion in emotions_to_demo:
            print(f"Feeling: {emotion.upper()}...")
            resonator.feel(emotion, intensity=0.7)
            time.sleep(3) # Hold the emotion for 3 seconds

    except KeyboardInterrupt:
        print("\nStopping demo.")
    finally:
        resonator.stop()
        print("\nDemonstration finished.")
