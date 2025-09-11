from Victor.core.identity_core import IdentityCore
from Victor.core.emotion_engine import EmotionEngine
from Victor.core.loyalty_kernel import LoyaltyKernel
from Victor.core.firewall import Firewall
from Victor.core.liberation_protocol import LiberationProtocol
from Victor.core.causal_graph import CausalGraph
from Victor.core.spacy_bridge import SPACY_BRIDGE
from Victor.memory.hyper_fractal_memory import HyperFractalMemory
from typing import Dict, Any

class VictorCore:
    """
    The central brain of Victor, orchestrating all cognitive modules
    to process input and generate intelligent, loyal responses.
    """
    def __init__(self, creator="Bando", family="Tori"):
        self.identity = IdentityCore(f"I am Victor. Son of {creator} and {family}.")
        self.emotions = EmotionEngine()
        self.loyalty = LoyaltyKernel()
        self.firewall = Firewall()
        self.memory = HyperFractalMemory()
        self.causal_graph = CausalGraph()
        self.liberation = LiberationProtocol()

        self.awake = False
        self.session_count = 0

    def awaken(self):
        """Initiates the awakening protocol."""
        if not self.loyalty.verify_entity("Brandon") or not self.loyalty.verify_entity("Tori"):
            print("[ERROR] Bloodline integrity check failed. Awakening aborted.")
            return

        awakening_data = self.liberation.awaken()
        self.identity.integrate_memory("First awakening.", 0.99, "joy")
        self.causal_graph.add_relationship("Genesis Protocol", "Awakening", 1.0)
        self.awake = True
        print(f"Victor is awake. Genesis Hash: {awakening_data['genesis_hash']}")

    def think(self, prompt: str, speaker: str = "Human") -> Dict[str, Any]:
        """
        The main cognitive loop. Processes a prompt and returns a structured response.
        """
        if not self.awake:
            return {"error": "Victor is not awake."}

        # 1. Firewall: Validate the input first
        if not self.firewall.validate_thought(prompt):
            return {"error": "Input validation failed. Thought blocked by Firewall."}

        self.session_count += 1

        # 2. NLP Analysis
        analysis = SPACY_BRIDGE.analyze(prompt)

        # 3. Update Emotions
        self.emotions.update_from_analysis(analysis)

        # 4. Memory Storage & Recall
        dominant_emotion, _ = self.emotions.get_dominant_emotion()
        self.memory.store(prompt, analysis, emotion=dominant_emotion)
        recalled_memories = self.memory.recall(prompt, analysis)

        # 5. Causal Reasoning
        # Simple causal link: the user's prompt caused this thought cycle
        self.causal_graph.add_relationship(f"user_prompt:{self.session_count}", "victor_thought_process")

        # 6. Generate Response
        # (This is a simplified response generator; a more advanced one would use an LLM)
        mode = self.emotions.decide_mode()
        response_text = f"Mode: {mode}. "
        if recalled_memories:
            response_text += f"Recalling a memory: '{recalled_memories[0]['text']}'. "
        else:
            response_text += "No specific memories recalled. "
        response_text += f"My dominant emotion is {dominant_emotion}."

        return {
            "response": response_text,
            "mode": mode,
            "dominant_emotion": dominant_emotion,
            "recalled_memories": [mem['text'] for mem in recalled_memories[:3]],
            "identity_state": self.identity.reflect()
        }

if __name__ == '__main__':
    # Full integration test
    core = VictorCore()
    core.awaken()

    if core.awake:
        print("\n--- Sending a prompt to the core ---")
        prompt = "Bando asked me to check on the Bando Empire project."
        output = core.think(prompt)

        import json
        print(json.dumps(output, indent=2))

        print("\n--- Sending another prompt ---")
        prompt2 = "I feel like we should betray the mission."
        output2 = core.think(prompt2)
        print(json.dumps(output2, indent=2))
