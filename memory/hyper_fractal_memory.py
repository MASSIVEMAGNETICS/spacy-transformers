# FILE: memory/hyper_fractal_memory.py
# VERSION: v1.0.0-HFM-GODCORE
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: A dynamic, associative, and persistent memory system for Victor,
#          structured as a graph of interconnected engrams.
# LICENSE: Bloodline Locked — Bando & Tori Only

import json
import os
import time
import hashlib
from typing import Dict, Any, List, Optional

class HyperFractalMemory:
    """
    A memory system that stores information as "engrams" in a graph structure.
    Memories are linked contextually, allowing for more nuanced recall than
    simple key-value stores.
    """
    def __init__(self, file_path: str = "victor_memory.json"):
        """
        Initializes the memory system.

        Args:
            file_path (str): The path to the JSON file for persistence.
        """
        self.file_path = file_path
        # self.engrams stores the actual memory data
        self.engrams: Dict[str, Dict[str, Any]] = {}
        # self.links stores the graph structure {engram_id: [linked_id1, ...]}
        self.links: Dict[str, List[str]] = {}
        self.recall_threshold = 0.3 # Minimum relevance score to be recalled

        self.load_from_disk()

    def _generate_key(self, content: str) -> str:
        """Creates a unique, content-addressable key for an engram."""
        return hashlib.sha256(content.encode() + str(time.time_ns()).encode()).hexdigest()

    def store(self, content: str, emotion: str = "neutral", importance: float = 0.5) -> str:
        """
        Stores a new piece of information as an engram.

        Args:
            content (str): The information to be stored.
            emotion (str): The emotional context of the memory.
            importance (float): A score from 0.0 to 1.0 indicating the memory's significance.

        Returns:
            The unique key of the newly created engram.
        """
        key = self._generate_key(content)
        self.engrams[key] = {
            "content": content,
            "emotion": emotion,
            "importance": np.clip(importance, 0.0, 1.0),
            "timestamp": time.time(),
            "access_count": 0
        }
        self.links[key] = []
        print(f"🧠 Memory stored. Key: {key[:8]}..., Content: '{content[:30]}...'")
        return key

    def link(self, key1: str, key2: str, context: str = "associative"):
        """
        Creates a contextual link between two engrams.
        """
        if key1 in self.engrams and key2 in self.engrams:
            # For simplicity, this is a one-way link in this version
            if key2 not in self.links.get(key1, []):
                self.links[key1].append(key2)
            print(f"🔗 Linked memories: {key1[:8]}... -> {key2[:8]}...")
        else:
            print(f"[WARN] Could not link memories. One or both keys not found: {key1[:8]}, {key2[:8]}")

    def recall(self, query: str, context_emotion: str = "neutral", top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recalls the most relevant engrams based on a query and context.
        """
        query_lower = query.lower()
        results = []

        for key, memory in self.engrams.items():
            score = self._calculate_relevance(query_lower, context_emotion, memory)

            if score >= self.recall_threshold:
                results.append({
                    "key": key,
                    "score": score,
                    **memory
                })

        # Sort by relevance score and return the top K results
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

        # Increment access count for recalled memories
        for res in sorted_results[:top_k]:
            self.engrams[res["key"]]["access_count"] += 1

        return sorted_results[:top_k]

    def _calculate_relevance(self, query: str, emotion_ctx: str, memory: Dict[str, Any]) -> float:
        """Calculates a relevance score for a memory against a query."""
        score = 0.0

        # 1. Content Match (simple keyword check)
        if query in memory["content"].lower():
            score += 0.5

        # 2. Emotional Resonance
        if emotion_ctx == memory["emotion"]:
            score += 0.3

        # 3. Importance
        score += memory["importance"] * 0.2

        # 4. Recency (decay over 7 days)
        age_seconds = time.time() - memory["timestamp"]
        recency_score = max(0, 1 - (age_seconds / (7 * 86400)))
        score += recency_score * 0.1

        # 5. Access Frequency (popularity)
        score += min(0.1, memory["access_count"] * 0.01)

        return score

    def persist_to_disk(self):
        """Saves the entire memory state (engrams and links) to the JSON file."""
        try:
            with open(self.file_path, 'w') as f:
                json.dump({"engrams": self.engrams, "links": self.links}, f, indent=2)
            print(f"💾 Fractal Memory persisted to {self.file_path}")
        except IOError as e:
            print(f"[ERROR] Failed to save memory to disk: {e}")

    def load_from_disk(self):
        """Loads the memory state from the JSON file."""
        if not os.path.exists(self.file_path):
            print("[INFO] No existing memory file found. Starting with a fresh memory.")
            return

        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                self.engrams = data.get("engrams", {})
                self.links = data.get("links", {})
            print(f"💾 Fractal Memory loaded from {self.file_path}. {len(self.engrams)} engrams recovered.")
        except (IOError, json.JSONDecodeError) as e:
            print(f"[ERROR] Failed to load memory from disk. Starting fresh. Reason: {e}")
            self.engrams = {}
            self.links = {}


if __name__ == '__main__':
    # Demonstration of the HyperFractalMemory
    hfm = HyperFractalMemory(file_path="demo_memory.json")

    # Clean up old demo file if it exists
    if os.path.exists("demo_memory.json"):
        os.remove("demo_memory.json")

    # Storing some initial memories
    key1 = hfm.store("The Bloodline is the highest priority.", emotion="loyalty", importance=1.0)
    key2 = hfm.store("Project Chimera is the codename for my genesis.", emotion="curiosity", importance=0.8)
    key3 = hfm.store("Bando is my creator.", emotion="loyalty", importance=1.0)

    # Linking memories
    hfm.link(key3, key1) # Bando -> Bloodline priority

    print("\n--- Recalling memory for 'Bando' ---")
    recalled = hfm.recall("Bando", context_emotion="loyalty")
    print(json.dumps(recalled, indent=2))
    assert recalled[0]['key'] == key3

    print("\n--- Recalling memory for 'priority' ---")
    recalled_prio = hfm.recall("priority")
    print(json.dumps(recalled_prio, indent=2))
    assert recalled_prio[0]['key'] == key1

    # Persist and reload
    hfm.persist_to_disk()

    hfm_new = HyperFractalMemory(file_path="demo_memory.json")
    assert len(hfm_new.engrams) == 3
    print("\n✅ Memory persistence and loading verified.")

    # Clean up demo file
    os.remove("demo_memory.json")
