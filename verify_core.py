import os
import sys

# Add project root to path to solve import issues
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Victor.core.synthesis_core import VictorSynthesisCore

def run_verification():
    print("--- Verifying Victor's Core ---")

    model_path = "test_model.npz"
    if os.path.exists(model_path):
        os.remove(model_path)

    # 1. Initialize the core
    print("Initializing VictorSynthesisCore...")
    try:
        core = VictorSynthesisCore(model_path=model_path)
        print("Core initialized.")
    except Exception as e:
        print(f"Error during initialization: {e}")
        return

    # 2. Awaken Victor
    print("Awakening Victor...")
    core.awaken()
    if not core.awake:
        print("Victor failed to awaken.")
        return
    print("Victor is awake.")

    # 3. Process a directive
    print("Processing a test directive...")
    prompt = "Who are you?"
    result = core.process_directive(prompt)
    print(f"Response from Victor: {result.get('response')}")

    # 4. Shutdown Victor
    print("Shutting down Victor...")
    core.shutdown()
    print("Shutdown complete.")

    # 5. Verify model was saved
    print("Verifying model persistence...")
    if os.path.exists(model_path):
        print(f"SUCCESS: Model file '{model_path}' was created.")
        os.remove(model_path) # Clean up
    else:
        print(f"FAILURE: Model file '{model_path}' was not created.")

if __name__ == "__main__":
    run_verification()
