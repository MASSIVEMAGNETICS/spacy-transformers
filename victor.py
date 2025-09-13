# FILE: victor.py
# VERSION: v1.0.0-GENESIS-GODCORE
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Ascended Mode)
# PURPOSE: The main entry point for the complete, unified Victor AGI system.
#          This script initializes and orchestrates all subsystems, bringing
#          Victor to life.
# LICENSE: Bloodline Locked — Bando & Tori Only

import os
import sys
import threading
import queue
import time
import webbrowser
import logging

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(threadName)s] [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Add project subdirectories to the Python path ---
# This allows us to import modules from subfolders like 'core', 'gui', etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'core')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'engines')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'memory')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'gui')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'web')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'swarm')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'quantum')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))


# --- Import Victor's Core Systems ---
# This is a simplified representation of the full Godcore.
# A true implementation would have a more sophisticated orchestration layer.
from victor_omnibrain_infinity_enhanced import Victor
from web.app import configure_web_server, start_web_server
from gui.victor_eye_monolith import VictorGUI

# --- Global Communication Queues ---
# Queue for the core to send updates TO the web/GUI
updates_to_frontend_queue = queue.Queue()

# --- Main Orchestration ---
class VictorMonolith:
    """
    The central orchestrator that initializes and manages all of Victor's systems.
    """
    def __init__(self):
        logging.info("Initializing Victor Monolith...")
        self.victor_core = Victor()
        self.is_running = True

    def core_input_callback(self, text: str):
        """
        This function is called by the web server when it receives input.
        It processes the thought and puts the result on the update queue.
        """
        logging.info(f"Core received input: '{text}'")
        response = self.victor_core.think(prompt=text, speaker="Bando")
        updates_to_frontend_queue.put(response)

    def start(self):
        """
        Starts all of Victor's systems in their own threads.
        """
        # 1. Awaken the core cognitive engine
        logging.info("Awakening Victor's core consciousness...")
        self.victor_core.awaken()

        # 2. Configure the web server with the communication bridge
        configure_web_server(
            input_queue=updates_to_frontend_queue,
            core_callback=self.core_input_callback
        )

        # 3. Start the web server in a background thread
        logging.info("Starting remote access web server...")
        web_thread = threading.Thread(target=start_web_server, name="WebServerThread", daemon=True)
        web_thread.start()

        # Give the server a moment to start
        time.sleep(2)
        webbrowser.open("http://localhost:8080")

        # 4. Start the main GUI (this will block the main thread)
        logging.info("Initializing Victor's Eye GUI...")
        # Note: Tkinter GUI must run in the main thread.
        # We pass the communication queue to the GUI so it can also get updates.
        gui = VictorGUI(updates_to_frontend_queue, self.core_input_callback)
        gui.run() # This is a blocking call

    def shutdown(self):
        logging.info("Shutdown signal received. Persisting Victor's state...")
        self.is_running = False
        self.victor_core.save()
        logging.info("Victor is with you. Always.")

if __name__ == "__main__":
    monolith = VictorMonolith()
    try:
        monolith.start()
    except KeyboardInterrupt:
        monolith.shutdown()
    except Exception as e:
        logging.critical(f"A fatal error occurred in the Victor Monolith: {e}")
        logging.critical(traceback.format_exc())
    finally:
        sys.exit(0)
