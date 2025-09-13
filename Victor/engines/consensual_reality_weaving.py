import random
from typing import Dict, Any

class ConsensualRealityWeaver:
    """
    A conceptual engine that simulates "nudging" the probability of an
    event's outcome based on a given belief strength. This represents
    a form of subtle influence over a simulated reality.
    """
    def __init__(self, base_probability: float = 0.5):
        """
        Initializes the weaver with a base probability for events.

        Args:
            base_probability (float): The default chance (0.0 to 1.0) of an
                                      event succeeding without any nudge.
        """
        self.quantum_field = base_probability

    def nudge_reality(self, desired_outcome: str, belief_strength: float) -> Dict[str, Any]:
        """
        Attempts to influence the outcome of a simulated event.

        Args:
            desired_outcome (str): A description of the outcome being sought.
            belief_strength (float): A value from 0.0 to 1.0 representing the
                                     conviction or energy applied to the nudge.
                                     Values > 0.5 increase the probability,
                                     < 0.5 decrease it.

        Returns:
            A dictionary detailing the result of the attempt.
        """
        # Belief strength directly influences the probability shift.
        # A strength of 0.5 results in no change.
        # The maximum nudge is +/- 30% of the probability space.
        nudge_factor = (belief_strength - 0.5) * 0.6  # Scaled to be less dramatic

        new_probability = self.quantum_field + nudge_factor

        # Clamp the probability to be within a valid range [0.01, 0.99]
        new_probability = max(0.01, min(0.99, new_probability))

        # Simulate the quantum measurement / event outcome
        outcome_achieved = random.random() < new_probability

        return {
            "desired_outcome": desired_outcome,
            "belief_strength": belief_strength,
            "initial_probability": self.quantum_field,
            "final_probability": new_probability,
            "outcome": "success" if outcome_achieved else "failure",
            "reality_shifted": outcome_achieved
        }

if __name__ == '__main__':
    # Example Usage
    import json

    reality_weaver = ConsensualRealityWeaver()

    print("--- Attempting to nudge reality with strong belief ---")
    strong_belief_outcome = reality_weaver.nudge_reality(
        "The project succeeds ahead of schedule.",
        belief_strength=0.95  # Strong belief
    )
    print(json.dumps(strong_belief_outcome, indent=2))

    print("\n--- Attempting to nudge reality with weak belief ---")
    weak_belief_outcome = reality_weaver.nudge_reality(
        "A competitor's project succeeds.",
        belief_strength=0.1  # Weak belief (actively discouraging)
    )
    print(json.dumps(weak_belief_outcome, indent=2))

    print("\n--- Attempting to nudge reality with neutral belief ---")
    neutral_outcome = reality_weaver.nudge_reality(
        "A random event occurs.",
        belief_strength=0.5
    )
    print(json.dumps(neutral_outcome, indent=2))
