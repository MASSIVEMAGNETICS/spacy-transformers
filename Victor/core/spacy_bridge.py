import spacy
from typing import List, Dict, Any, Tuple

class SpaCyBridge:
    """
    A bridge to the spaCy NLP library, providing a simple interface
    to process text and extract structured information.
    """
    _nlp = None

    def __init__(self):
        if SpaCyBridge._nlp is None:
            try:
                SpaCyBridge._nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Error: spaCy model 'en_core_web_sm' not found.")
                print("Please run 'python -m spacy download en_core_web_sm'")
                raise

    def process_text(self, text: str) -> spacy.tokens.doc.Doc:
        """Processes text using the loaded spaCy model."""
        return self._nlp(text)

    def extract_entities(self, doc: spacy.tokens.doc.Doc) -> List[Dict[str, str]]:
        """Extracts named entities from a processed spaCy Doc."""
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    def extract_pos_tags(self, doc: spacy.tokens.doc.Doc) -> List[Dict[str, str]]:
        """Extracts part-of-speech tags from a processed spaCy Doc."""
        return [{"text": token.text, "pos": token.pos_, "tag": token.tag_} for token in doc]

    def extract_noun_chunks(self, doc: spacy.tokens.doc.Doc) -> List[str]:
        """Extracts noun chunks from a processed spaCy Doc."""
        return [chunk.text for chunk in doc.noun_chunks]

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Performs a full analysis of the text, returning a dictionary
        of extracted information.
        """
        doc = self.process_text(text)
        return {
            "text": text,
            "entities": self.extract_entities(doc),
            "pos_tags": self.extract_pos_tags(doc),
            "noun_chunks": self.extract_noun_chunks(doc),
        }

# Global instance for easy access
SPACY_BRIDGE = SpaCyBridge()

if __name__ == '__main__':
    # Example usage:
    text_to_analyze = "Brandon and Tori are building the Bando Empire in Lorain, Ohio."
    analysis_result = SPACY_BRIDGE.analyze(text_to_analyze)

    import json
    print(json.dumps(analysis_result, indent=2))
