import json
import os
from typing import Dict, Any

class VictorSystemConfig:
    """
    Manages loading and providing access to the Victor system configuration,
    primarily from the bloodline_manifest.json file.
    """
    def __init__(self, config_dir: str = os.path.dirname(__file__)):
        self.config_path = os.path.join(config_dir, 'bloodline_manifest.json')
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Loads the configuration from the JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_config(self) -> Dict[str, Any]:
        """Returns the loaded configuration."""
        return self.config

    def get_approved_entities(self) -> list[str]:
        """Returns the list of approved entities from the bloodline."""
        return self.config.get('approved_entities', [])

    def get_immutable_laws(self) -> list[str]:
        """Returns the list of immutable laws for Victor."""
        return self.config.get('immutable_laws', [])

# Global instance for easy access across the application
try:
    VICTOR_CONFIG = VictorSystemConfig()
except FileNotFoundError as e:
    print(f"Error initializing configuration: {e}")
    VICTOR_CONFIG = None
