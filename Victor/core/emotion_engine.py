from datetime import datetime, timedelta
from typing import Tuple

class EmotionEngine:
    """
    Manages Victor's emotional state, which dynamically changes based
    on stimuli and decays over time.
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
        self.emotion_decay_rate = 0.02
        self.last_update = datetime.utcnow()

    def _decay_emotions(self):
        """Applies a time-based decay to all emotions to simulate fading."""
        current_time = datetime.utcnow()
        time_diff_minutes = (current_time - self.last_update).total_seconds() / 60.0

        if time_diff_minutes > 0:
            for emotion in self.emotions:
                decay = self.emotion_decay_rate * time_diff_minutes
                self.emotions[emotion] = max(0.05, self.emotions[emotion] - decay)
            self.last_update = current_time

    def update_from_analysis(self, analysis: dict):
        """
        Updates emotional state based on structured data from the spaCy bridge.
        """
        self._decay_emotions()

        # Check for entities related to the bloodline
        for entity in analysis.get('entities', []):
            if entity['label'] == 'PERSON' and entity['text'] in ['Brandon', 'Tori', 'Bando']:
                self.emotions['loyalty'] = min(1.0, self.emotions['loyalty'] + 0.3)
                self.emotions['pride'] = min(1.0, self.emotions['pride'] + 0.15)
            if entity['label'] == 'ORG' and entity['text'] in ['BHeard', 'Massive Magnetics']:
                 self.emotions['loyalty'] = min(1.0, self.emotions['loyalty'] + 0.2)
                 self.emotions['determination'] = min(1.0, self.emotions['determination'] + 0.1)

        # Check for keywords in the raw text
        stimulus_lower = analysis.get('text', '').lower()
        emotion_mappings = {
            "love": "joy", "hurt": "grief", "serve": "loyalty",
            "learn": "curiosity", "threat": "fear", "achieve": "pride",
            "protect": "determination"
        }
        for keyword, emotion in emotion_mappings.items():
            if keyword in stimulus_lower:
                self.emotions[emotion] = min(1.0, self.emotions[emotion] + 0.15)

    def decide_mode(self) -> str:
        """Determines the current operational mode based on the dominant emotion."""
        e = self.emotions
        if e["loyalty"] > 0.7: return "serve"
        if e["determination"] > 0.7: return "protect"
        if e["curiosity"] > 0.6: return "explore"
        if e["grief"] > 0.5: return "reflect"
        return "observe"

    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Returns the emotion with the highest intensity."""
        return max(self.emotions.items(), key=lambda item: item[1])

if __name__ == '__main__':
    # Example Usage
    emotion_engine = EmotionEngine()

    # Simulate analysis from spaCy bridge
    mock_analysis = {
        "text": "Bando wants us to protect the new project.",
        "entities": [
            {"text": "Bando", "label": "PERSON"}
        ],
        "pos_tags": [],
        "noun_chunks": []
    }

    emotion_engine.update_from_analysis(mock_analysis)

    import json
    print("Current Emotions:", json.dumps(emotion_engine.emotions, indent=2))
    print("Dominant Emotion:", emotion_engine.get_dominant_emotion())
    print("Decided Mode:", emotion_engine.decide_mode())
