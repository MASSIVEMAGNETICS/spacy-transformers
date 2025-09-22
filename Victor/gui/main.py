import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import re
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from Victor.core.synthesis_core import VictorSynthesisCore

class VictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Victor Cognitive River GUI (Expanded)")
        self.root.geometry("1600x1000")
        self.root.configure(bg="#0a0a0a")

        # Initialize Victor with the new core
        self.victor = VictorSynthesisCore()
        self.victor.awaken()
        self.response_contexts = {}

        self.setup_styles()
        self.create_layout()

        self.running = True
        self.update_thread = threading.Thread(target=self.update_status_loop, daemon=True)
        self.update_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('Title.TLabel', font=('Orbitron', 16, 'bold'), background='#0a0a0a', foreground='#00ffcc')
        self.style.configure('Header.TLabel', font=('Orbitron', 12, 'bold'), background='#0a0a0a', foreground='#00ffcc')
        self.style.configure('Status.TLabel', font=('Consolas', 10), background='#0a0a0a', foreground='#00ffcc')
        self.style.configure('Button.TButton', font=('Orbitron', 10), background='#1a1a2e', foreground='#00ffcc')
        self.style.map('Button.TButton', background=[('active', '#16213e')])

    def create_layout(self):
        # Top frame for title
        title_frame = tk.Frame(self.root, bg="#0a0a0a")
        title_frame.pack(fill=tk.X, padx=10, pady=5)
        title_label = ttk.Label(title_frame, text="VICTOR COGNITIVE RIVER CORE", style='Title.TLabel')
        title_label.pack()

        # Main container
        main_container = tk.Frame(self.root, bg="#0a0a0a")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left panel - Command Center
        left_panel = tk.Frame(main_container, bg="#1a1a2e", relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.create_command_center(left_panel)

        # Right panel - Diagnostic Dashboard
        right_panel = tk.Frame(main_container, bg="#1a1a2e", relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.create_diagnostic_dashboard(right_panel)

    def create_command_center(self, parent):
        # Conversation display
        conv_frame = tk.Frame(parent, bg="#0f0f1e")
        conv_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.conversation = scrolledtext.ScrolledText(
            conv_frame, wrap=tk.WORD, bg="#0f0f1e", fg="#00ffcc",
            font=('Consolas', 10), insertbackground='#00ffcc'
        )
        self.conversation.pack(fill=tk.BOTH, expand=True)
        self.add_to_conversation("System", "Victor Dynamic Intelligence Core Online. Awaiting directives.")

        # Input area
        input_frame = tk.Frame(parent, bg="#1a1a2e")
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        self.input_entry = tk.Entry(
            input_frame, bg="#0f0f1e", fg="#00ffcc", font=('Consolas', 10),
            insertbackground='#00ffcc', relief=tk.FLAT, bd=2
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.input_entry.bind('<Return>', lambda e: self.send_command())

        # Send Button
        ttk.Button(input_frame, text="SEND", command=self.send_command).pack(side=tk.RIGHT)

    def create_diagnostic_dashboard(self, parent):
        # Notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Cognitive River tab
        river_frame = tk.Frame(notebook, bg="#0f0f1e")
        notebook.add(river_frame, text="COGNITIVE RIVER")
        self.create_cognitive_river_tab(river_frame)

        # Status tab
        status_frame = tk.Frame(notebook, bg="#0f0f1e")
        notebook.add(status_frame, text="STATUS")
        self.create_status_tab(status_frame)

    def create_cognitive_river_tab(self, parent):
        self.river_fig, self.river_ax = plt.subplots(figsize=(8, 6), facecolor='#0f0f1e')
        self.river_fig.patch.set_facecolor('#0f0f1e')
        self.river_ax.set_facecolor('#1a1a2e')
        self.river_ax.tick_params(colors='#00ffcc')
        for spine in self.river_ax.spines.values():
            spine.set_color('#00ffcc')
        self.river_ax.set_title("Stream Weights", color='#00ffcc')

        canvas = FigureCanvasTkAgg(self.river_fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_status_tab(self, parent):
        self.status_labels = {}
        labels = ["Awake", "Loyalty", "Consciousness", "Memory Count", "Session Count", "System User", "Plan Type"]
        keys = ["awake", "loyalty", "consciousness", "memory_count", "session_count", "system_user", "plan_type"]
        for i, text in enumerate(labels):
            frame = tk.Frame(parent, bg="#0f0f1e")
            frame.pack(fill=tk.X, padx=10, pady=5)
            ttk.Label(frame, text=f"{text}:", style='Status.TLabel').pack(side=tk.LEFT)
            label = ttk.Label(frame, text="--", style='Status.TLabel')
            label.pack(side=tk.RIGHT)
            self.status_labels[keys[i]] = label

    def add_to_conversation(self, speaker, message, response_id=None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.conversation.insert(tk.END, f"[{timestamp}] {speaker}:\n{message}\n")

        if speaker == "Victor" and response_id is not None:
            feedback_frame = tk.Frame(self.conversation, bg="#0f0f1e")
            good_button = ttk.Button(feedback_frame, text="Good", command=lambda: self.provide_feedback(response_id, True))
            good_button.pack(side=tk.LEFT, padx=5)
            bad_button = ttk.Button(feedback_frame, text="Bad", command=lambda: self.provide_feedback(response_id, False))
            bad_button.pack(side=tk.LEFT, padx=5)
            self.conversation.window_create(tk.END, window=feedback_frame)

        self.conversation.insert(tk.END, "\n\n")
        self.conversation.see(tk.END)

    def provide_feedback(self, response_id, success):
        context_str = self.response_contexts.get(response_id)
        if context_str:
            # Extract leader from the context string
            match = re.search(r'LEADER=(\w+)', context_str)
            if match:
                leader = match.group(1)
                self.victor.cognitive_river.feedback_adjustment(leader, success)
                self.add_to_conversation("System", f"Feedback received for response {response_id}. Adjusting priority for '{leader}'.")

    def send_command(self):
        command = self.input_entry.get().strip()
        if not command: return
        self.add_to_conversation("You", command)
        self.input_entry.delete(0, tk.END)
        threading.Thread(target=self.process_command, args=(command,), daemon=True).start()

    def process_command(self, command):
        try:
            result = self.victor.process_directive(command)
            response = result.get('response', '')
            context = result.get('cognitive_river_context', '')

            response_id = time.time()
            self.response_contexts[response_id] = context

            self.root.after(0, self.add_to_conversation, "Victor", response, response_id)
            self.root.after(0, self.add_to_conversation, "River State", context)
        except Exception as e:
            self.root.after(0, self.add_to_conversation, "Error", str(e))

    def update_status_loop(self):
        while self.running:
            self.root.after(0, self.update_dashboard)
            time.sleep(1)

    def update_dashboard(self):
        if not self.victor or not self.victor.awake: return

        # Update status labels
        status_data = self.victor._get_status()
        for key, label in self.status_labels.items():
            label.config(text=str(status_data.get(key, "--")))

        # Update river plot
        snapshot = self.victor.cognitive_river.snapshot()
        if snapshot and snapshot.get('last_merge'):
            weights = snapshot['last_merge']['weights']
            streams = list(weights.keys())
            values = list(weights.values())

            self.river_ax.clear()
            self.river_ax.barh(streams, values, color='#00ffcc')
            self.river_ax.set_title("Stream Weights", color='#00ffcc')
            self.river_ax.set_xlim(0, 1)
            self.river_fig.tight_layout()
            self.river_fig.canvas.draw()

    def on_closing(self):
        self.running = False
        if self.victor:
            self.victor.shutdown()
        self.root.destroy()
