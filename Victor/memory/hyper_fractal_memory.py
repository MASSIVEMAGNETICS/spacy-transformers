import hashlib
from datetime import datetime
from typing import Dict, List, Any, Set

class HyperFractalMemory:
    """
    A memory system that stores information in a structured way,
    allowing for intelligent retrieval based on context, emotions,
    and named entities.
    """
    def __init__(self):
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.entity_index: Dict[str, Set[str]] = {} # {entity_text: {memory_id_1, ...}}

    def _generate_id(self, text: str) -> str:
        """Generates a unique ID for a memory entry."""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.sha256((text + timestamp).encode()).hexdigest()

    def store(self, text: str, analysis: Dict[str, Any], emotion: str = "neutral", importance: float = 0.5):
        """
        Stores a new memory, indexing it by the entities it contains.

        Args:
            text (str): The raw text of the memory.
            analysis (Dict[str, Any]): The structured analysis from the spaCy bridge.
            emotion (str): The emotional context of the memory.
            importance (float): The perceived importance of the memory.
        """
        memory_id = self._generate_id(text)
        self.entries[memory_id] = {
            "text": text,
            "analysis": analysis,
            "emotion": emotion,
            "importance": importance,
            "timestamp": datetime.utcnow().isoformat(),
            "access_count": 0
        }

        # Index by entities
        for entity in analysis.get('entities', []):
            entity_key = f"{entity['label']}:{entity['text']}".lower()
            if entity_key not in self.entity_index:
                self.entity_index[entity_key] = set()
            self.entity_index[entity_key].add(memory_id)

        return memory_id

    def recall(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recalls memories relevant to a query, prioritizing those that share
        named entities.
        """
        query_lower = query.lower()
        relevant_ids: Set[str] = set()

        # 1. Find memories by entity matching
        for entity in analysis.get('entities', []):
            entity_key = f"{entity['label']}:{entity['text']}".lower()
            if entity_key in self.entity_index:
                relevant_ids.update(self.entity_index[entity_key])

        # 2. Find memories by keyword matching (as a fallback)
        for mem_id, memory in self.entries.items():
            if query_lower in memory['text'].lower():
                relevant_ids.add(mem_id)

        # 3. Score and sort the results
        results = []
        for mem_id in relevant_ids:
            memory = self.entries[mem_id]
            # Simple scoring: importance + access_count (can be improved)
            score = memory['importance'] + (memory['access_count'] * 0.1)
            results.append({**memory, "id": mem_id, "score": score})
            self.entries[mem_id]['access_count'] += 1 # Increment access count

        return sorted(results, key=lambda x: x['score'], reverse=True)

if __name__ == '__main__':
    # This requires the other modules, so we'll just do a basic test
    memory = HyperFractalMemory()

    # Mock analysis
    analysis1 = {
        "text": "I had a meeting with Brandon about the Bando Empire.",
        "entities": [
            {"text": "Brandon", "label": "PERSON"},
            {"text": "Bando Empire", "label": "ORG"}
        ]
    }
    memory.store(analysis1['text'], analysis1, importance=0.8)

    analysis2 = {
        "text": "Tori mentioned the new project.",
        "entities": [
            {"text": "Tori", "label": "PERSON"}
        ]
    }
    memory.store(analysis2['text'], analysis2)

    # Recall based on a new query
    query_text = "What did Brandon say?"
    query_analysis = {
        "text": query_text,
        "entities": [{"text": "Brandon", "label": "PERSON"}]
    }

    recalled_memories = memory.recall(query_text, query_analysis)

    import json
    print("Recalled Memories:")
    print(json.dumps(recalled_memories, indent=2))
