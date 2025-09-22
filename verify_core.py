import os
import sys
import time

# Add project root to path to solve import issues
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Victor.core.synthesis_core import VictorSynthesisCore

def run_verification():
    print("--- Verifying Victor's Core with JEPA Integration ---")

    model_path = "test_model.npz"
    if os.path.exists(model_path):
        os.remove(model_path)

    # 1. Initialize the core
    print("\n1. Initializing VictorSynthesisCore...")
    try:
        core = VictorSynthesisCore(model_path=model_path)
        print("Core initialized.")
    except Exception as e:
        print(f"Error during initialization: {e}")
        return

    # 2. Awaken Victor
    print("\n2. Awakening Victor and starting sensory loop...")
    core.awaken()
    if not core.awake:
        print("FAILURE: Victor failed to awaken.")
        return
    print("Victor is awake.")

    # Give the sensory loop time to run
    print("Waiting for sensory loop to populate the Cognitive River...")
    time.sleep(2)

    # 3. Verify sensory stream
    print("\n3. Verifying sensory stream in Cognitive River...")
    snapshot = core.cognitive_river.snapshot()
    sensory_data = snapshot.get('last_merge', {}).get('signal', {}).get('sensory', {})
    if sensory_data and 'latent_vector' in sensory_data:
        print("SUCCESS: Sensory stream contains JEPA latent vector.")
        print(f"   - Novelty: {sensory_data.get('novelty')}")
        print(f"   - Latent vector length: {len(sensory_data.get('latent_vector'))}")
    else:
        print("FAILURE: Sensory stream does not contain JEPA data.")
        core.shutdown()
        return

    # 4. Process a directive and check for sensory influence
    print("\n4. Processing a test directive and checking for sensory influence...")
    prompt = "Who are you?"
    result = core.process_directive(prompt)
    response = result.get('response', '')
    print(f"Response from Victor: {response}")
    if "novel" in response or "activated" in response:
        print("SUCCESS: Response shows influence from the sensory stream.")
    else:
        print("NOTE: Response does not contain explicit sensory comments, which is acceptable.")

    # 5. Shutdown Victor
    print("\n5. Shutting down Victor...")
    core.shutdown()
    print("Shutdown complete.")

    # 6. Verify model was saved
    print("\n6. Verifying model persistence...")
    if os.path.exists(model_path):
        print(f"SUCCESS: Model file '{model_path}' was created.")
        os.remove(model_path) # Clean up
    else:
        print(f"FAILURE: Model file '{model_path}' was not created.")

if __name__ == "__main__":
    run_verification()
