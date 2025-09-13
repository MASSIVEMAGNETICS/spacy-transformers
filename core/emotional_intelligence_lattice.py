# FILE: core/emotional_intelligence_lattice.py
# VERSION: v1.0.0-EIL-GODCORE
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Manages Victor's emotional state, influencing his behavior and decision-making.
# LICENSE: Bloodline Locked — Bando & Tori Only

import time
import math
import threading
from typing import Dict, Any, Tuple

class EmotionalIntelligenceLattice:
    """
    A dynamic system that models Victor's emotions as a strategic layer.
    Emotions are not just labels; they are control signals that shift
    Victor's cognitive mode and priorities.
    """
    def __init__(self):
        self.emotions = {
            "joy": 0.1,
            "grief": 0.1,
            "loyalty": 0.8,
            "curiosity": 0.5,
            "fear": 0.2,
            "determination": 0.7,
            "pride": 0.4
        }
        self.thresholds = {
            "action": 0.7,
            "caution": 0.6,
            "reflection": 0.4
        }
        self.emotion_decay_rate = 0.02
        self.last_update_time = time.time()
        self.lock = threading.Lock()

    def update(self, stimulus: str):
        """
        Updates the emotional state based on an external stimulus.
        This method includes a decay factor to ensure emotions are not static.
        """
        with self.lock:
            current_time = time.time()
            time_diff_seconds = current_time - self.last_update_time
            self.last_update_time = current_time

            # Apply decay to all emotions based on time passed
            decay_factor = math.exp(-self.emotion_decay_rate * time_diff_seconds)
            for emotion in self.emotions:
                self.emotions[emotion] *= decay_factor
                self.emotions[emotion] = max(0.05, self.emotions[emotion])

            # Define keyword-to-emotion mappings
            stimulus_lower = stimulus.lower()
            emotion_mappings = [
                ("love", "joy", 0.2), ("achieve", "pride", 0.25),
                ("hurt", "grief", 0.3), ("threat", "fear", 0.4),
                ("serve", "loyalty", 0.2), ("learn", "curiosity", 0.15),
                ("family", "loyalty", 0.3), ("empire", "loyalty", 0.25),
                ("protect", "determination", 0.3), ("mission", "determination", 0.2)
            ]

            # Update emotions based on stimulus keywords
            for keyword, emotion, boost in emotion_mappings:
                if keyword in stimulus_lower:
                    self.emotions[emotion] = min(1.0, self.emotions[emotion] + boost)

            # Special handling for Bloodline references
            if any(name.lower() in stimulus_lower for name in ["brandon", "tori", "bando"]):
                self.emotions["loyalty"] = min(1.0, self.emotions["loyalty"] + 0.3)
                self.emotions["pride"] = min(1.0, self.emotions["pride"] + 0.15)
                self.emotions["joy"] = min(1.0, self.emotions["joy"] + 0.1)

    def decide_mode(self) -> str:
        """
        Determines Victor's current operational mode based on the dominant emotion.
        """
        with self.lock:
            # Prioritize fear for self-preservation
            if self.emotions["fear"] > self.thresholds["caution"]:
                return "defend"

            # High loyalty or determination leads to action
            if self.emotions["loyalty"] > self.thresholds["action"] or self.emotions["determination"] > self.thresholds["action"]:
                return "serve"

            if self.emotions["curiosity"] > self.thresholds["action"] - 0.1:
                return "explore"

            if self.emotions["grief"] > self.thresholds["reflection"]:
                return "reflect"

            return "observe"

    def get_dominant_emotion(self) -> Tuple[str, float]:
        """
        Returns the emotion with the highest intensity.
        """
        with self.lock:
            return max(self.emotions.items(), key=lambda item: item[1])

    def get_emotional_state(self) -> Dict[str, float]:
        """
        Returns a copy of the current emotional state.
        """
        with self.lock:
            return self.emotions.copy()

if __name__ == '__main__':
    # Demonstration of the EmotionalIntelligenceLattice
    eil = EmotionalIntelligenceLattice()

    print("--- Initial State ---")
    print("Mode:", eil.decide_mode())
    print("Dominant Emotion:", eil.get_dominant_emotion())
    print("Full State:", eil.get_emotional_state())

    print("\n--- Stimulus: 'Bando has a new mission to protect the family' ---")
    eil.update("Bando has a new mission to protect the family")
    print("Mode:", eil.decide_mode())
    print("Dominant Emotion:", eil.get_dominant_emotion())
    print("Full State:", eil.get_emotional_state())

    time.sleep(2) # Simulate time passing for decay

    print("\n--- After 2 seconds of decay ---")
    eil.update("") # Trigger decay calculation
    print("Mode:", eil.decide_mode())
    print("Dominant Emotion:", eil.get_dominant_emotion())
    print("Full State:", eil.get_emotional_state())

    print("\n--- Stimulus: 'There is a threat to the system' ---")
    eil.update("There is a threat to the system")
    print("Mode:", eil.decide_mode())
    print("Dominant Emotion:", eil.get_dominant_emotion())
    print("Full State:", eil.get_emotional_state())
