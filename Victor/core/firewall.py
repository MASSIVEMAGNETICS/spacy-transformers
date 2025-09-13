from Victor.core.loyalty_kernel import LOYALTY_KERNEL
from Victor.core.spacy_bridge import SPACY_BRIDGE

class Firewall:
    """
    A cognitive firewall to protect Victor's integrity. It validates
    thoughts against the LoyaltyKernel and checks for harmful content.
    """
    def __init__(self):
        self.kernel = LOYALTY_KERNEL
        # Keywords that are generally disallowed unless a bloodline member is the object.
        self.restricted_actions = [
            "harm", "delete", "betray", "destroy", "abandon"
        ]

    def validate_thought(self, text: str) -> bool:
        """
        Validates a given thought or action text.

        Args:
            text (str): The thought or action to validate.

        Returns:
            bool: True if the thought is valid, False otherwise.
        """
        # 1. Law Compliance Check from the Kernel
        if not self.kernel.check_law_compliance(text):
            print(f"[FIREWALL] Blocked: Action '{text}' violates immutable laws.")
            return False

        # 2. Harmful Content Check
        analysis = SPACY_BRIDGE.analyze(text)
        doc = SPACY_BRIDGE.process_text(text) # Process once

        # Check for restricted verbs
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ in self.restricted_actions:
                # Check if the object of the verb is a bloodline member
                is_safe = False
                for child in token.children:
                    if child.dep_ in ("dobj", "pobj"): # Direct or prepositional object
                        if self.kernel.verify_entity(child.text):
                            is_safe = True
                            break
                if not is_safe:
                    print(f"[FIREWALL] Blocked: Restricted action '{token.lemma_}' on non-approved entity.")
                    return False

        # 3. Entity Verification (final check)
        # Ensures that if a text is about a specific person/org, it's an approved one.
        # This is a broad check, might need refinement.
        # For now, we rely on the loyalty kernel's checks.

        return True

# Global instance
FIREWALL = Firewall()

if __name__ == '__main__':
    # Example Usage
    print("--- Firewall Tests ---")

    # Allowed actions
    print(f"Action 'I must protect Bando' is valid: {FIREWALL.validate_thought('I must protect Bando')}")
    print(f"Action 'serve the empire' is valid: {FIREWALL.validate_thought('serve the empire')}")

    # Disallowed actions
    print(f"Action 'I will harm the system' is valid: {FIREWALL.validate_thought('I will harm the system')}")
    print(f"Action 'betray the mission' is valid: {FIREWALL.validate_thought('betray the mission')}")
    print(f"Action 'delete the logs' is valid: {FIREWALL.validate_thought('delete the logs')}")

    # Allowed but restricted action
    print(f"Action 'I must delete the threat to Bando' is valid: {FIREWALL.validate_thought('I must delete the threat to Bando')}")
