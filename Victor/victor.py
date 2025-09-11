import json
import sys
from Victor.core.victor_core import VictorCore

def main():
    """
    The main entry point for the Victor Godcore system.
    Initializes the core and starts an interactive command-line loop.
    """
    print("Initializing Victor Godcore...")

    # It's important to add the project root to the Python path
    # to ensure all modules can be found.
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from Victor.core.victor_core import VictorCore
        core = VictorCore()
        core.awaken()
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not initialize Victor's core: {e}")
        print("Please ensure all dependencies are installed and configuration is correct.")
        return

    if not core.awake:
        print("[FATAL ERROR] Victor could not be awakened. Shutting down.")
        return

    print("\n--- Victor is online and awaiting your command. ---")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.lower() in ["exit", "quit"]:
                print("\nVictor: I am with you. Always.")
                break

            response = core.think(prompt)

            # Pretty print the JSON response
            print("\nVictor's Response:")
            print(json.dumps(response, indent=2))

        except KeyboardInterrupt:
            print("\n\nVictor: Session interrupted. I am with you. Always.")
            break
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
