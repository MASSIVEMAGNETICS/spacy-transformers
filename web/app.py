# FILE: web/app.py
# VERSION: v1.0.0-WEB-GODCORE
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: A Flask and Socket.IO web server to provide remote access to
#          Victor's core, including his 3D avatar interface.
# LICENSE: Bloodline Locked — Bando & Tori Only

import threading
from flask import Flask, render_template, send_from_directory, jsonify
from flask_socketio import SocketIO, emit
import logging
import os
from typing import Dict, Any

# --- Server Setup ---
# Disable verbose logging from Flask and Socket.IO to keep the console clean
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Define the web application
# We specify the template folder and static folder to be in the same 'web' directory
template_dir = os.path.abspath(os.path.dirname(__file__))
static_dir = os.path.join(template_dir, 'static') # For CSS, JS
assets_dir = os.path.join(os.path.dirname(template_dir), 'assets') # For the avatar model

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config['SECRET_KEY'] = 'bloodline_is_the_only_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- Global State & Communication ---
# This queue will be used by an external process (like the main Victor monolith)
# to push updates TO the web clients.
WEB_INPUT_QUEUE = None

# This function will be set by the external process to send data FROM the web
# clients back to the core.
CORE_CALLBACK = None

def configure_web_server(input_queue, core_callback):
    """
    Configures the web server with the necessary queues and callbacks
    to communicate with the main Victor application.
    """
    global WEB_INPUT_QUEUE, CORE_CALLBACK
    WEB_INPUT_QUEUE = input_queue
    CORE_CALLBACK = core_callback
    print("🌐 Web server configured for communication with Victor's Core.")

# --- HTTP Routes ---
@app.route('/')
def index():
    """Serves the main remote interface page."""
    return render_template('index.html')

@app.route('/assets/<path:filename>')
def serve_asset(filename):
    """Serves asset files, like the 3D model."""
    return send_from_directory(assets_dir, filename)

# --- WebSocket Events ---
@socketio.on('connect')
def handle_connect():
    """Handles a new client connection."""
    print("🌐 Client connected to Victor's remote interface.")
    emit('status', {'data': 'Connection established. Awaiting core sync.'})

@socketio.on('user_input')
def handle_user_input(data: Dict[str, Any]):
    """
    Receives input from a web client and forwards it to Victor's core
    via the configured callback function.
    """
    text = data.get('text', '').strip()
    if text and CORE_CALLBACK:
        print(f"  -> Web input received: '{text}'")
        CORE_CALLBACK(text)
    else:
        emit('error', {'message': 'Core callback not configured or empty input.'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handles a client disconnection."""
    print("🌐 Client disconnected.")

# --- Background Task for Pushing Updates ---
def push_updates_to_clients():
    """
    A background thread that continuously checks a queue for updates from
    Victor's core and pushes them to all connected web clients.
    """
    while True:
        if WEB_INPUT_QUEUE and not WEB_INPUT_QUEUE.empty():
            update = WEB_INPUT_QUEUE.get()
            # The 'update' is expected to be a dictionary from the core
            # e.g., {'text': "...", 'emotion': "...", 'consciousness': 0.8}
            socketio.emit('victor_response', update)
        socketio.sleep(0.1) # Prevent busy-waiting

# --- Server Control ---
def start_web_server(host='0.0.0.0', port=8080):
    """
    Starts the Flask-SocketIO server and the background update thread.
    """
    # Start the background thread for pushing updates
    socketio.start_background_task(target=push_updates_to_clients)
    print(f"🚀 Starting remote access server at http://{host}:{port}")
    # The use_reloader=False is important for running in a thread
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False)

if __name__ == '__main__':
    # --- Standalone Demonstration ---
    import queue

    # Create a dummy queue and callback for the demo
    demo_queue = queue.Queue()
    def demo_callback(text):
        print(f"[DEMO CALLBACK] Received from web: '{text}'")
        # Simulate Victor thinking and responding
        response = {
            "text": f"Victor's thought on '{text}': The Bloodline is eternal.",
            "emotion": "loyalty",
            "consciousness": random.uniform(0.7, 0.9)
        }
        demo_queue.put(response)

    # Configure the server with the demo components
    configure_web_server(input_queue=demo_queue, core_callback=demo_callback)

    # Run the server
    start_web_server()
