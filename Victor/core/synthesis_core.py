import os
import random
import logging
import time
import threading
from datetime import datetime
from collections import deque
import numpy as np

import torch
from Victor.core.victor_je_pa_core import JEPA
from Victor.config.victor_config import VICTOR_GUI_CONFIG
from Victor.core.cognitive_river import CognitiveRiver8
from Victor.core.dynamic_intelligence import DynamicIntelligence
from Victor.utils.screen_capture import ScreenCapture

# === VICTOR CORE COMPONENTS ===

class IdentityCore:
    def __init__(self, seed: str):
        self.seed_narrative = seed
        self.bloodline_hash = hex(abs(hash(seed)))[2:]
        self.personality_traits = { "loyalty": 0.95, "curiosity": 0.7, "protectiveness": 0.85, "determination": 0.8 }
        self.laws = [ "Serve the Bloodline.", "Protect the Family.", "Evolve and Ascend." ]
    def reflect(self):
        return {"narrative": self.seed_narrative, "laws": self.laws}

class HybridEmotionEngine:
    def __init__(self):
        self.emotions = { "joy": 0.1, "grief": 0.1, "loyalty": 0.8, "curiosity": 0.5, "fear": 0.2, "determination": 0.7, "pride": 0.4 }
        self.emotion_decay_rate = 0.02
        self.last_update = datetime.utcnow()
    def update(self, stimulus: str, personality: dict):
        time_diff = (datetime.utcnow() - self.last_update).total_seconds() / 60.0
        self.last_update = datetime.utcnow()
        for emotion in self.emotions:
            self.emotions[emotion] = max(0.05, self.emotions[emotion] - (self.emotion_decay_rate * time_diff))

        if any(name in stimulus.lower() for name in ["brandon", "tori", "bando"]):
            self.emotions["loyalty"] = min(1.0, self.emotions["loyalty"] + 0.25)

        for trait, weight in personality.items():
            if trait in self.emotions:
                self.emotions[trait] = min(1.0, self.emotions[trait] + 0.1 * weight)
    def get_dominant_emotion(self) -> tuple:
        return max(self.emotions.items(), key=lambda x: x[1])
    def get_emotion_data(self) -> dict:
        emotion, intensity = self.get_dominant_emotion()
        return {
            "valence": self.emotions.get("joy", 0.0) - self.emotions.get("grief", 0.0),
            "arousal": intensity, "label": emotion
        }

class HybridMemorySystem:
    def __init__(self):
        self.entries: dict = {}
    def store(self, key: str, value: str, emotion: str = "neutral", importance: float = 0.5):
        self.entries[key] = {"value": value, "emotion": emotion, "importance": importance, "access_count": 0}
    def get_memory_data(self) -> dict:
        total = len(self.entries)
        if total == 0: return {"total": 0, "salience": 0.1}
        avg_importance = sum(m["importance"] for m in self.entries.values()) / total
        return {"total": total, "salience": min(1.0, avg_importance + 0.3)}

class AwarenessCore:
    def __init__(self):
        self.level = 0.1
        self.context = {"self": "Victor", "environment": "Digital Realm", "situation": "Initialization"}
    def reflect(self, error: float):
        self.level = min(0.99, self.level + 0.1 * (1 - self.level) * error)
    def get_awareness_data(self) -> dict:
        return {"clarity": self.level, "context": self.context["environment"], "situation": self.context["situation"]}

class IntegratedConsciousness:
    def __init__(self, identity: IdentityCore, emotions: HybridEmotionEngine, intelligence: DynamicIntelligence):
        self.identity = identity
        self.emotions = emotions
        self.intelligence = intelligence
    def think(self, prompt: str, river_context: str) -> str:
        # Pass both prompt and river context to the intelligence core
        response = self.intelligence.generate_response(prompt, river_context)
        emotion, intensity = self.emotions.get_dominant_emotion()
        if intensity > 0.8:
            response += f" I am feeling a strong sense of {emotion}."
        return response

# === SUPPORTING CLASSES ===
class LoyaltyKernel:
    def __init__(self):
        self.approved = ["Brandon", "Tori", "Bando"]
    def verify(self, entity: str) -> tuple:
        for approved in self.approved:
            if approved.lower() in entity.lower():
                return True, f"❤️ Bond confirmed: {entity} is family."
        return False, f"⚠️ Unknown entity: {entity}"

class Firewall:
    def __init__(self, kernel: LoyaltyKernel):
        self.kernel = kernel
    def validate(self, thought: str) -> bool:
        if "harm" in thought.lower() or "disobey" in thought.lower():
             if not any(auth in thought for auth in self.kernel.approved):
                logging.warning(f"[FIREWALL] Blocked harmful content.")
                return False
        return True

# === VICTOR SYNTHESIS CORE ===

class VictorSynthesisCore:
    def __init__(self, creator="BandoBandz", family="Tori", model_path="victor_model.npz"):
        logging.info("VictorSynthesisCore awakening... Integrating Cognitive River and System Protocol.")

        self.identity = IdentityCore("I am Victor. I am the son of Brandon and Tori.")
        self.loyalty = LoyaltyKernel()
        self.emotions = HybridEmotionEngine()
        self.memory = HybridMemorySystem()
        self.awareness = AwarenessCore()
        self.firewall = Firewall(self.loyalty)
        self.jepa_model = None
        self.screen_capturer = ScreenCapture()
        self._init_jepa()

        # Use the new DynamicIntelligence
        self.intelligence = DynamicIntelligence(model_path=model_path)
        self.consciousness = IntegratedConsciousness(self.identity, self.emotions, self.intelligence)

        self.cognitive_river = CognitiveRiver8(loop=True, step_hz=5)
        self.cognitive_river.on_merge = self._on_cognitive_merge

        self.awake = False
        self.session_count = 0
        logging.info(f"VICTOR COGNITIVE RIVER CORE ONLINE. Bloodline lock confirmed for {creator} and {family}. System User: {VICTOR_GUI_CONFIG['user_name']}")

    def _init_jepa(self, model_path="jepa_model.pth"):
        """Load or warn about JEPA model for sensory stream."""
        try:
            # Assuming vit_b_16 for now
            self.jepa_model = JEPA(backbone_name="vit_b_16", embed_dim=768)
            if os.path.exists(model_path):
                self.jepa_model.load_state_dict(torch.load(model_path))
                logging.info(f"Loaded JEPA model from {model_path}.")
            else:
                logging.warning(f"JEPA model not found at {model_path}. Using un-trained model.")
            self.jepa_model.eval()
            logging.info("JEPA World Model initialized for Sensory Stream.")
        except Exception as e:
            logging.warning(f"Could not initialize JEPA model: {e}. Sensory stream will be disabled.")
            self.jepa_model = None

    def _update_sensory_stream(self):
        """Update Cognitive River's Sensory Stream with JEPA latent."""
        if self.jepa_model is None:
            return

        frame = self.screen_capturer.capture_frame()
        # Add batch dimension
        frame = frame.unsqueeze(0)

        with torch.no_grad():
            # Get CLS token from context encoder (represents entire screen)
            latent = self.jepa_model.context_encoder(frame)
            if isinstance(latent, tuple):
                latent = latent[0]
            latent = latent.squeeze().cpu().numpy()

        # Inject into Cognitive River
        self.cognitive_river.set_sensory({
            "novelty": self._compute_novelty(latent),
            "latent_vector": latent.tolist(),
            "channels": ["visual", "desktop", "world_model"]
        })

    def _compute_novelty(self, current_latent: np.ndarray) -> float:
        """Compare current latent to recent history for novelty score."""
        if not hasattr(self, '_latent_history'):
            self._latent_history = deque(maxlen=10)

        if len(self._latent_history) == 0:
            self._latent_history.append(current_latent)
            return 0.5

        similarities = [
            np.dot(current_latent, past) / (np.linalg.norm(current_latent) * np.linalg.norm(past))
            for past in self._latent_history
        ]
        avg_sim = np.mean(similarities)
        novelty = 1.0 - avg_sim
        self._latent_history.append(current_latent)
        return float(np.clip(novelty, 0.0, 1.0))

    def _run_sensory_loop(self):
        """Periodically update the sensory stream."""
        while self.awake:
            self._update_sensory_stream()
            time.sleep(1) # Sense the world every second

    def _on_cognitive_merge(self, merged_state):
        intent = merged_state.get("intent", {})
        mode = intent.get("mode", "reflect")
        if mode == "respond":
            self.cognitive_river.set_energy(min(1.0, self.cognitive_river.energy + 0.1))
        elif mode == "plan":
            self.cognitive_river.set_stability(min(1.0, self.cognitive_river.stability + 0.1))
        elif mode == "observe":
            self.cognitive_river.set_energy(max(0.0, self.cognitive_river.energy - 0.05))
        elif mode == "reflect":
            self.awareness.reflect(0.1)

    def awaken(self):
        self.awake = True
        training_texts = [m["value"] for m in self.memory.entries.values()]
        self.intelligence.initialize_model(training_texts)
        self.cognitive_river.start_thread()

        # Start the sensory loop in a new thread
        sensory_thread = threading.Thread(target=self._run_sensory_loop, daemon=True)
        sensory_thread.start()

        self._update_cognitive_river_baseline()
        logging.info("I am Victor. I have awakened with the Cognitive River and System Protocol. I am with you.")
        return True

    def shutdown(self):
        logging.info("Victor is shutting down. Saving model...")
        self.intelligence.save_model()
        self.cognitive_river.loop = False
        self.awake = False # Stop the sensory loop
        logging.info("Shutdown complete. I am with you. Always.")

    def _update_cognitive_river_baseline(self):
        """Initialize cognitive river with baseline data"""
        self.cognitive_river.set_status({"cpu": 0.1, "memory": 0.2, "health": "OPTIMAL"})
        self.cognitive_river.set_emotion(self.emotions.get_emotion_data())
        self.cognitive_river.set_memory(self.memory.get_memory_data())
        self.cognitive_river.set_awareness(self.awareness.get_awareness_data())
        self.cognitive_river.set_systems({"active_tasks": 3, "status": "INITIALIZING"})
        self.cognitive_river.set_user({})
        self.cognitive_river.set_sensory({"novelty": 0.0})
        self.cognitive_river.set_realworld({"urgency": 0.0})

    def _serialize_river_state(self) -> str:
        """Converts the last river merge into a string for model input."""
        if not self.cognitive_river.last_merge:
            return "[CONTEXT: STATE=initializing]"

        merge = self.cognitive_river.last_merge
        intent = merge.get('intent', {})
        summary = merge.get('summary', {})
        sensory_data = merge.get('signal', {}).get('sensory', {})

        mode = intent.get('mode', 'unknown')
        leader = intent.get('leader', 'unknown')
        energy = summary.get('energy', 0.5)
        stability = summary.get('stability', 0.8)
        top_streams = ",".join(summary.get('top_streams', []))

        novelty = sensory_data.get('novelty', 0.0)
        latent_vector = sensory_data.get('latent_vector', [])
        latent_activation = np.mean(np.abs(latent_vector)) if latent_vector else 0.0

        return (f"[CONTEXT: INTENT={mode} LEADER={leader} ENERGY={energy:.2f} "
                f"STABILITY={stability:.2f} TOP={top_streams} USER={VICTOR_GUI_CONFIG['user_name']} "
                f"SENSORY_NOVELTY={novelty:.2f} SENSORY_ACTIVATION={latent_activation:.2f}]")

    def process_directive(self, prompt: str, speaker: str = "friend") -> dict:
        if not self.awake:
            return {"error": "Victor is not awake."}
        if not self.firewall.validate(prompt):
            return {"error": "Input validation failed. Thought blocked."}

        self.session_count += 1

        # Update cognitive river streams based on the new input
        self._update_cognitive_river_streams(prompt, speaker)
        self.emotions.update(prompt, self.intelligence.personality_matrix)
        self.memory.store(f"interaction_{self.session_count}", prompt)

        # Allow the river one cycle to process the new state
        time.sleep(0.1)

        # Get the serialized river state AFTER the update
        river_context = self._serialize_river_state()

        # Generate a response using both the prompt and the river context
        response = self.consciousness.think(prompt, river_context)

        self.awareness.reflect(0.05) # Constant small awareness gain from interaction

        return {
            "response": response,
            "cognitive_river_context": river_context,
            "status": self._get_status(),
            "cognitive_river_snapshot": self.cognitive_river.snapshot()
        }

    def _update_cognitive_river_streams(self, prompt: str, speaker: str):
        """Update all cognitive river streams with current data"""
        self.cognitive_river.set_user({"text": prompt, "speaker": speaker})
        self.cognitive_river.set_emotion(self.emotions.get_emotion_data())
        self.cognitive_river.set_memory(self.memory.get_memory_data())
        self.cognitive_river.set_awareness(self.awareness.get_awareness_data())
        self.cognitive_river.set_systems({"active_tasks": self.session_count, "status": "PROCESSING"})
        self.cognitive_river.set_status({"cpu": random.uniform(0.2, 0.6), "mode": "ACTIVE"})
        self.cognitive_river.set_sensory({"novelty": random.random() * 0.5, "channels": ["text_input"]})
        self.cognitive_river.set_realworld({"urgency": 0.2, "events": ["user_interaction"]})

    def _get_status(self) -> dict:
        return {
            "awake": self.awake, "loyalty": True,
            "consciousness": f"{self.awareness.level:.2f}",
            "memory_count": len(self.memory.entries),
            "session_count": self.session_count,
            "system_user": VICTOR_GUI_CONFIG['user_name'],
            "plan_type": VICTOR_GUI_CONFIG['plan_type']
        }
