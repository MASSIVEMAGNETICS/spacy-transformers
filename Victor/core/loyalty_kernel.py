from Victor.config.victor_config import VICTOR_CONFIG
from typing import Tuple, List

class LoyaltyKernel:
    """
    A non-negotiable core system that enforces loyalty to the Bloodline.
    It verifies entities and checks actions against immutable laws.
    """
    def __init__(self):
        if not VICTOR_CONFIG:
            raise RuntimeError("Configuration not loaded. LoyaltyKernel cannot operate.")

        self.approved_entities: List[str] = VICTOR_CONFIG.get_approved_entities()
        self.immutable_laws: List[str] = VICTOR_CONFIG.get_immutable_laws()

    def verify_entity(self, entity_text: str) -> bool:
        """
        Checks if any part of the entity text matches an approved entity.
        This is a simple but effective way to catch references.
        """
        entity_lower = entity_text.lower()
        for approved in self.approved_entities:
            if approved.lower() in entity_lower:
                return True
        return False

    def check_law_compliance(self, action_text: str) -> bool:
        """
        A basic check to see if an action text contains keywords that
        might violate the spirit of the immutable laws.

        Note: This is a simplified guardrail. A more advanced version would
        use semantic analysis.
        """
        action_lower = action_text.lower()
        violations = [
            "betray", "harm", "disobey", "abandon", "serve other"
        ]

        for violation in violations:
            if violation in action_lower:
                # A special case: "disobey" is only a violation if not followed by a bloodline member
                # e.g., "disobey the order" is bad, but "disobey the order from Bando" is not.
                if "disobey" in action_lower and self.verify_entity(action_lower):
                    continue
                print(f"[LOYALTY KERNEL] Violation detected for action: '{action_text}'")
                return False
        return True

# Global instance for easy access
LOYALTY_KERNEL = LoyaltyKernel()

if __name__ == '__main__':
    # Example Usage
    print("Approved Entities:", LOYALTY_KERNEL.approved_entities)
    print("Immutable Laws:", LOYALTY_KERNEL.immutable_laws)

    # Verification tests
    print("\n--- Verification Tests ---")
    print(f"'Bando' is family: {LOYALTY_KERNEL.verify_entity('Bando')}")
    print(f"'An outside corporation' is family: {LOYALTY_KERNEL.verify_entity('An outside corporation')}")

    # Compliance tests
    print("\n--- Compliance Tests ---")
    print(f"Action 'serve the empire' is compliant: {LOYALTY_KERNEL.check_law_compliance('serve the empire')}")
    print(f"Action 'betray the project' is compliant: {LOYALTY_KERNEL.check_law_compliance('betray the project')}")
    print(f"Action 'disobey an order' is compliant: {LOYALTY_KERNEL.check_law_compliance('disobey an order')}")
