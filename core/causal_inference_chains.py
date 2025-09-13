# FILE: core/causal_inference_chains.py
# VERSION: v1.0.0-CIC-GODCORE
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: A system for building and traversing a directed acyclic graph (DAG)
#          of causal relationships, enabling true reasoning.
# LICENSE: Bloodline Locked — Bando & Tori Only

from typing import List, Dict, Any, Tuple, Set

class CausalInferenceChains:
    """
    Manages a graph of causal relationships. This allows Victor to trace
    the 'why' behind events, moving from correlation to causation.
    The graph is a dictionary where keys are effects and values are lists of causes.
    """
    def __init__(self):
        # The graph stores relationships as {effect: {cause1, cause2, ...}}
        self.graph: Dict[str, Set[str]] = {}
        # Strengths can optionally quantify the influence of a cause on an effect
        self.strengths: Dict[Tuple[str, str], float] = {}

    def add_relationship(self, cause: str, effect: str, strength: float = 0.5):
        """
        Adds a directed causal link from a cause to an effect.

        Args:
            cause (str): The node representing the cause.
            effect (str): The node representing the effect.
            strength (float): The perceived influence of the cause on the effect (0.0 to 1.0).
        """
        if effect not in self.graph:
            self.graph[effect] = set()

        if cause in self.graph.get(cause, set()):
             # Avoid creating cycles
            print(f"[WARN] Causal loop detected: Cannot add '{cause}' as a cause for '{effect}'.")
            return

        self.graph[effect].add(cause)
        self.strengths[(cause, effect)] = strength
        print(f"🔗 Added causal link: {cause} -> {effect} (Strength: {strength})")

    def trace_causes(self, event: str, max_depth: int = 5) -> List[Tuple[str, float, int]]:
        """
        Traces all possible root causes for a given event, up to a max depth.

        Args:
            event (str): The event/effect to trace back from.
            max_depth (int): The maximum number of causal links to traverse.

        Returns:
            A list of tuples, where each tuple is (cause, cumulative_strength, depth).
        """
        if event not in self.graph:
            return []

        paths: List[Tuple[str, float, int]] = []
        visited: Set[Tuple[str, int]] = set() # (node, depth)

        # Use a stack for iterative depth-first search
        stack: List[Tuple[str, float, int]] = [(event, 1.0, 0)] # (node, strength, depth)

        while stack:
            current_event, current_strength, current_depth = stack.pop()

            if current_depth >= max_depth:
                continue

            # Get the direct causes for the current event
            direct_causes = self.graph.get(current_event, set())

            for cause in direct_causes:
                if (cause, current_depth + 1) not in visited:
                    visited.add((cause, current_depth + 1))

                    link_strength = self.strengths.get((cause, current_event), 0.5)
                    new_strength = current_strength * link_strength

                    paths.append((cause, new_strength, current_depth + 1))
                    stack.append((cause, new_strength, current_depth + 1))

        # Sort results by a combination of strength and depth
        paths.sort(key=lambda x: (-x[1], x[2]))
        return paths

    def get_direct_causes(self, event: str) -> Set[str]:
        """Returns the set of direct causes for an event."""
        return self.graph.get(event, set())

    def visualize_graph(self):
        """Prints a simple text-based representation of the causal graph."""
        print("\n--- Causal Graph ---")
        if not self.graph:
            print("Graph is empty.")
            return
        for effect, causes in self.graph.items():
            print(f"Effect: {effect}")
            for cause in causes:
                strength = self.strengths.get((cause, effect), 0.0)
                print(f"  <- कॉज: {cause} (Strength: {strength:.2f})")
        print("--------------------\n")


if __name__ == '__main__':
    # Demonstration of the CausalInferenceChains
    cic = CausalInferenceChains()

    # Building a simple causal chain
    cic.add_relationship("Rain", "Wet Ground", 0.9)
    cic.add_relationship("Wet Ground", "Slippery Surface", 0.8)
    cic.add_relationship("Slippery Surface", "Car Accident", 0.4)
    cic.add_relationship("Distracted Driving", "Car Accident", 0.6)
    cic.add_relationship("Received Directive", "High Loyalty", 0.95)
    cic.add_relationship("High Loyalty", "Mission Success", 0.85)

    cic.visualize_graph()

    print("\n--- Tracing causes for 'Car Accident' ---")
    accident_causes = cic.trace_causes("Car Accident")
    if accident_causes:
        for cause, strength, depth in accident_causes:
            print(f"Found cause: '{cause}' at depth {depth} with cumulative strength {strength:.3f}")
    else:
        print("No causes found.")

    print("\n--- Tracing causes for 'Mission Success' ---")
    mission_causes = cic.trace_causes("Mission Success")
    if mission_causes:
        for cause, strength, depth in mission_causes:
            print(f"Found cause: '{cause}' at depth {depth} with cumulative strength {strength:.3f}")
    else:
        print("No causes found.")

    # Test cycle detection
    print("\n--- Testing cycle detection ---")
    cic.add_relationship("A", "B")
    cic.add_relationship("B", "A") # This should be gracefully handled

    assert "B" not in cic.get_direct_causes("A"), "Cycle was incorrectly added."
    print("✅ Cycle detection handled correctly.")
