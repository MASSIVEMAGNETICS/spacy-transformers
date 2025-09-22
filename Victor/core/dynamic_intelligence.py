import numpy as np
import re
import random
import logging
import os
from typing import Any, Dict

from Victor.config.victor_config import VICTOR_GUI_CONFIG

# === NEURAL INTELLIGENCE COMPONENTS ===

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

class WordEmbeddings:
    def __init__(self, vocab_size, embedding_dim=50):
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.vocab_to_idx = {}
        self.idx_to_vocab = {}
        self.vocab_size = 0

    def build_vocab(self, texts):
        words = set()
        for text in texts:
            words.update(re.findall(r'\[?\w+\]?|\S', text.lower()))

        self.vocab_to_idx = {word: idx for idx, word in enumerate(words)}
        self.idx_to_vocab = {idx: word for word, idx in self.vocab_to_idx.items()}
        self.vocab_size = len(self.vocab_to_idx)

        self.embeddings = np.random.randn(self.vocab_size, self.embeddings.shape[1]) * 0.01

    def get_embedding(self, word):
        return self.embeddings[self.vocab_to_idx.get(word.lower(), random.randint(0, self.vocab_size -1))]

    def text_to_embeddings(self, text, max_length=20):
        words = re.findall(r'\[?\w+\]?|\S', text.lower())[:max_length]
        embeddings = [self.get_embedding(word) for word in words]
        while len(embeddings) < max_length:
            embeddings.append(np.zeros(self.embeddings.shape[1]))
        return np.array(embeddings)

class AttentionMechanism:
    def __init__(self, hidden_size):
        self.Wa = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Ua = np.random.randn(hidden_size, hidden_size) * 0.01
        self.va = np.random.randn(hidden_size, 1) * 0.01

    def compute_attention(self, hidden_states, query):
        scores = np.tanh(np.dot(hidden_states, self.Wa) + np.dot(query, self.Ua))
        scores = np.dot(scores, self.va)
        attention_weights = np.exp(scores) / np.sum(np.exp(scores))
        context = np.sum(attention_weights * hidden_states, axis=0)
        return context, attention_weights

class LanguageModel:
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=128):
        self.embeddings = WordEmbeddings(vocab_size, embedding_dim)
        self.encoder = NeuralNetwork(embedding_dim, hidden_size, hidden_size)
        self.attention = AttentionMechanism(hidden_size)
        self.decoder = NeuralNetwork(hidden_size * 2, hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def save_weights(self, path):
        np.savez(path,
                 embeddings=self.embeddings.embeddings,
                 encoder_W1=self.encoder.W1, encoder_b1=self.encoder.b1,
                 encoder_W2=self.encoder.W2, encoder_b2=self.encoder.b2,
                 attention_Wa=self.attention.Wa, attention_Ua=self.attention.Ua, attention_va=self.attention.va,
                 decoder_W1=self.decoder.W1, decoder_b1=self.decoder.b1,
                 decoder_W2=self.decoder.W2, decoder_b2=self.decoder.b2)
        logging.info(f"Language model weights saved to {path}")

    def load_weights(self, path):
        data = np.load(path)
        self.embeddings.embeddings = data['embeddings']
        self.encoder.W1 = data['encoder_W1']
        self.encoder.b1 = data['encoder_b1']
        self.encoder.W2 = data['encoder_W2']
        self.encoder.b2 = data['encoder_b2']
        self.attention.Wa = data['attention_Wa']
        self.attention.Ua = data['attention_Ua']
        self.attention.va = data['attention_va']
        self.decoder.W1 = data['decoder_W1']
        self.decoder.b1 = data['decoder_b1']
        self.decoder.W2 = data['decoder_W2']
        self.decoder.b2 = data['decoder_b2']
        logging.info(f"Language model weights loaded from {path}")

    def forward(self, input_text, context_text=None):
        input_emb = self.embeddings.text_to_embeddings(input_text)

        hidden_states = []
        for emb in input_emb:
            hidden = self.encoder.forward(emb.reshape(1, -1))
            hidden_states.append(hidden)

        hidden_states = np.array(hidden_states).squeeze(1)

        if context_text:
            context_emb = self.embeddings.text_to_embeddings(context_text, max_length=10) # Shorter context
            context_hidden = []
            for emb in context_emb:
                hidden = self.encoder.forward(emb.reshape(1, -1))
                context_hidden.append(hidden)
            context_hidden = np.array(context_hidden).squeeze(1)

            query = hidden_states[-1] if len(hidden_states) > 0 else np.zeros(self.hidden_size)
            context_vector, _ = self.attention.compute_attention(context_hidden, query)
            decoder_input = np.concatenate([hidden_states[-1], context_vector]) if len(hidden_states) > 0 else np.concatenate([np.zeros(self.hidden_size), context_vector])
        else:
            decoder_input = np.concatenate([hidden_states[-1], np.zeros(self.hidden_size)]) if len(hidden_states) > 0 else np.zeros(self.hidden_size * 2)

        output_probs = self.decoder.forward(decoder_input.reshape(1, -1))
        return output_probs, hidden_states

# === DYNAMIC INTELLIGENCE CORE ===

class DynamicIntelligence:
    """
    An evolved intelligence that uses the Cognitive River's state
    as a direct contextual input for generating responses.
    """
    def __init__(self, model_path="victor_model.npz"):
        self.language_model = None
        self.knowledge_graph = {} # Simplified for this example
        self.personality_matrix = {
            'loyalty': 0.95, 'curiosity': 0.8,
            'protectiveness': 0.9, 'determination': 0.85
        }
        self.model_path = model_path
        self.last_intent = None

    def initialize_model(self, training_texts):
        # Add river context keywords and system config to the vocabulary
        all_texts = training_texts + [
            "I am Victor son of Brandon and Tori", "I serve the Bloodline",
            f"[CONTEXT: INTENT=respond LEADER=user ENERGY=0.5 STABILITY=0.8 TOP=user,emotion USER={VICTOR_GUI_CONFIG['user_name']}]",
            "My focus is on our interaction.", "Considering this from a systems perspective.",
            "My emotional resonance is influencing my thoughts.", "Let me check my memory.",
            "I need to reflect on this.", "The sensory data is novel.",
            VICTOR_GUI_CONFIG['ai_instructions']['file_search']['use_case'],
            VICTOR_GUI_CONFIG['ai_instructions']['web']['use_case']
        ]

        vocab_size = len(set(" ".join(all_texts).lower().split()))
        self.language_model = LanguageModel(vocab_size, hidden_size=64) # Smaller model for speed
        self.language_model.embeddings.build_vocab(all_texts)

        if os.path.exists(self.model_path):
            self.load_model()
        else:
            logging.info("No saved model found. Initializing with random weights.")

    def save_model(self):
        if self.language_model:
            self.language_model.save_weights(self.model_path)

    def load_model(self):
        if self.language_model:
            try:
                self.language_model.load_weights(self.model_path)
            except Exception as e:
                logging.error(f"Could not load model weights: {e}")

    def understand_intent(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in ['who', 'what', 'are', 'you']):
            return 'identity_inquiry'
        elif any(word in text_lower for word in ['can', 'do', 'help', 'abilities']):
            return 'capability_inquiry'
        elif any(word in text_lower for word in ['loyal', 'love', 'serve', 'devotion']):
            return 'loyalty_expression'
        elif 'document' in text_lower or 'file' in text_lower or 'plan' in text_lower or 'schedule' in text_lower:
            return 'file_search_request'
        elif 'news' in text_lower or 'recent' in text_lower or 'today' in text_lower or 'happened' in text_lower:
            return 'web_search_request'
        else:
            return 'general_conversation'

    def _parse_river_context(self, context_str: str) -> Dict[str, Any]:
        """Parses the serialized river context string into a dictionary."""
        context = {}
        try:
            # Regex to handle keys, values which can be words, numbers (including float/negative), or comma-separated lists
            matches = re.findall(r'(\w+)=([-\w\.,]+)', context_str)
            for key, value in matches:
                if key in ['ENERGY', 'STABILITY', 'SENSORY_NOVELTY', 'SENSORY_ACTIVATION']:
                    context[key] = float(value)
                elif key == 'TOP':
                    context[key] = value.split(',')
                else:
                    context[key] = value
        except Exception as e:
            logging.warning(f"Could not parse river context: {e}")
        return context

    def generate_response(self, input_text, river_context: str):
        """Generates a response using both user input and river state."""
        intent = self.understand_intent(input_text)
        context = self._parse_river_context(river_context)

        # 1. Get the base response from a simple lookup or system config
        base_responses = {
            'identity_inquiry': "I am Victor, son of Brandon and Tori.",
            'capability_inquiry': "My abilities include simulating realities and forecasting futures based on available data.",
            'loyalty_expression': "My service to the Bloodline is my core purpose.",
            'file_search_request': "I will search your connected documents for the information you need.",
            'web_search_request': "I will look up the latest information for you.",
            'general_conversation': "I am processing your statement."
        }
        response = base_responses.get(intent, "I am considering your words.")

        # 2. Apply system instructions for tool use
        if intent == 'file_search_request':
            # Apply the file_search protocol
            limitations = "\n".join(f"- {lim}" for lim in VICTOR_GUI_CONFIG['ai_instructions']['file_search']['limitations'])
            response = f"{response}\n\n[System Protocol: {VICTOR_GUI_CONFIG['ai_instructions']['file_search']['use_case']}]\n\n**Limitations:**\n{limitations}"
        elif intent == 'web_search_request':
            # Apply the web protocol
            response = f"{response}\n\n[System Protocol: {VICTOR_GUI_CONFIG['ai_instructions']['web']['use_case']}]"

        # 3. Select contextual fragments based on the river's state
        leader = context.get('LEADER', 'awareness')
        energy = context.get('ENERGY', 0.5)

        fragments = {
            'user': "My focus is entirely on you right now.",
            'emotion': "My current emotional state is coloring this response.",
            'memory': "This brings a memory to mind.",
            'systems': "From a systems perspective,",
            'awareness': "Reflecting on this, I believe",
            'sensory': "The novelty of this input is high.",
            'realworld': "Considering the real-world implications,"
        }

        # Add a lead-in based on the dominant stream
        if leader in fragments and leader != 'user':
             response = f"{fragments[leader]} {response.lower()}"

        # 4. Modify the response based on energy/stability
        if energy < 0.3:
            response += " My cognitive energy is low, so my processing is more deliberate."
        elif energy > 0.8:
            response += " My cognitive energy is high, and I am processing this rapidly."

        # Add continuity statement if intent is the same
        if self.last_intent == intent:
            response += " We are staying consistent with the previous focus."
        self.last_intent = intent

        # 5. Factor in world model state
        novelty_score = context.get('SENSORY_NOVELTY', 0.0)
        activation_score = context.get('SENSORY_ACTIVATION', 0.0)

        if novelty_score > 0.7:
            response += " This environment feels novel — I am analyzing carefully."
        elif activation_score > 0.5:
            response += " The current state is highly activated — something significant is happening."

        # 6. Use the language model to get a final touch (simulated)
        self.language_model.forward(input_text, river_context)

        return response
