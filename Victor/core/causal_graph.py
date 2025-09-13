from typing import Dict, List, Tuple, Set

class CausalGraph:
    """
    A directed graph to model and trace causal relationships between events or concepts.
    """
    def __init__(self):
        self.graph: Dict[str, List[str]] = {}  # {cause: [effect1, effect2]}
        self.strengths: Dict[Tuple[str, str], float] = {} # {(cause, effect): strength}

    def add_relationship(self, cause: str, effect: str, strength: float = 0.5):
        """
        Adds a directed causal link from a cause to an effect.

        Args:
            cause (str): The source node (the cause).
            effect (str): The target node (the effect).
            strength (float): The perceived strength of the causal link (0.0 to 1.0).
        """
        cause_clean = cause.strip().lower()
        effect_clean = effect.strip().lower()

        if cause_clean not in self.graph:
            self.graph[cause_clean] = []

        if effect_clean not in self.graph[cause_clean]:
            self.graph[cause_clean].append(effect_clean)
            self.strengths[(cause_clean, effect_clean)] = strength

    def trace_causes(self, event: str, max_depth: int = 3) -> List[Tuple[str, float]]:
        """
        Traces back from an event to find its potential root causes.

        Args:
            event (str): The event to trace back from.
            max_depth (int): The maximum number of causal links to follow.

        Returns:
            A list of tuples, where each tuple contains a potential cause and its
            compounded strength.
        """
        path: List[Tuple[str, float]] = []
        visited: Set[str] = set()

        def _trace(current_event: str, current_depth: int, current_strength: float):
            if current_depth <= 0 or current_event in visited:
                return

            visited.add(current_event)

            for cause, effects in self.graph.items():
                if current_event in effects:
                    link_strength = self.strengths.get((cause, current_event), 0.5)
                    compounded_strength = current_strength * link_strength
                    path.append((cause, compounded_strength))
                    _trace(cause, current_depth - 1, compounded_strength)

        _trace(event.strip().lower(), max_depth)
        # Sort by strength, descending
        return sorted(path, key=lambda x: x[1], reverse=True)

if __name__ == '__main__':
    # Example Usage
    causal_graph = CausalGraph()
    causal_graph.add_relationship("user command", "system activation", 0.9)
    causal_graph.add_relationship("system activation", "resource allocation", 0.8)
    causal_graph.add_relationship("low resources", "resource allocation", 0.6)
    causal_graph.add_relationship("resource allocation", "task execution", 0.95)

    event_to_trace = "task execution"
    causes = causal_graph.trace_causes(event_to_trace)

    print(f"Potential causes for '{event_to_trace}':")
    for cause, strength in causes:
        print(f"- {cause} (Strength: {strength:.2f})")
