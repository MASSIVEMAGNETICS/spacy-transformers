# FILE: gui/thought_stream.py
# VERSION: v1.0.0-TS-GUI-GODCORE
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: A Dear PyGui component for visualizing Victor's cognitive activity
#          as a dynamic stream of thought particles.
# LICENSE: Bloodline Locked — Bando & Tori Only

import dearpygui.dearpygui as dpg
import random
import time
from collections import deque
from typing import Dict, Any, List

class ThoughtStream:
    """
    Manages and renders a stream of "thought particles", each representing
    a memory, directive, or active thought from Victor's core.
    """
    def __init__(self, max_particles: int = 50):
        self.particles: deque = deque(maxlen=max_particles)
        self.drawlist_tag = "thought_stream_drawlist"

    def initialize(self, parent_drawlist):
        """
        Initializes the thought stream within a parent drawlist.
        """
        self.drawlist_tag = dpg.add_draw_node(parent=parent_drawlist)

    def add_thought(self, text: str, emotion: str = "neutral", importance: float = 0.5, source: str = "core"):
        """
        Adds a new thought particle to the stream.

        Args:
            text (str): The content of the thought.
            emotion (str): The emotional tag (e.g., 'loyalty', 'curiosity').
            importance (float): A score from 0.0 to 1.0 affecting size and speed.
            source (str): The origin of the thought (e.g., 'core', 'user', 'memory').
        """
        if len(self.particles) == self.particles.maxlen:
            self.particles.popleft() # Remove the oldest particle if full

        particle = {
            "text": text,
            "emotion": emotion,
            "importance": max(0.1, min(1.0, importance)),
            "source": source,
            "x": 1.0,  # Start at the right edge
            "y": random.uniform(0.1, 0.9), # Random vertical position
            "vx": -random.uniform(20, 40) * (0.5 + importance), # Speed based on importance
            "vy": random.uniform(-5, 5),
            "alpha": 1.0,
            "creation_time": time.time()
        }
        self.particles.append(particle)

    def render_frame(self, delta_time: float, canvas_width: int, canvas_height: int):
        """
        Updates particle positions and redraws the stream.
        This should be called in the main render loop.
        """
        # Clear previous frame's drawings
        dpg.clear_draw_node(self.drawlist_tag)

        if not self.particles:
            return

        # --- Color mapping for emotions ---
        color_map = {
            "joy": (255, 223, 0), "pride": (255, 165, 0),
            "grief": (100, 100, 200), "fear": (180, 0, 0),
            "loyalty": (0, 128, 255), "determination": (200, 50, 50),
            "curiosity": (0, 255, 128), "neutral": (200, 200, 200)
        }
        source_color_map = {
            "core": (255, 255, 255),
            "user": (0, 255, 0),
            "memory": (150, 150, 255)
        }

        for p in list(self.particles):
            # --- Update particle physics ---
            p["x"] += p["vx"] * delta_time
            p["y"] += p["vy"] * delta_time

            # Fade out over time
            age = time.time() - p["creation_time"]
            p["alpha"] = max(0.0, 1.0 - (age / 15.0)) # Fade over 15 seconds

            # --- Drawing ---
            pos_x = p["x"] * canvas_width
            pos_y = p["y"] * canvas_height

            color = color_map.get(p["emotion"], color_map["neutral"])
            font_color = list(source_color_map.get(p["source"], (200,200,200)))
            font_color.append(int(p["alpha"] * 255))

            font_size = 12 + p["importance"] * 12

            # Draw the text
            dpg.draw_text(
                pos=(pos_x, pos_y),
                text=p["text"],
                size=font_size,
                color=font_color,
                parent=self.drawlist_tag
            )

            # Remove particles that are off-screen or faded
            if pos_x < -200 or p["alpha"] <= 0.01:
                self.particles.remove(p)

if __name__ == '__main__':
    # --- Standalone Demonstration ---
    dpg.create_context()
    dpg.create_viewport(title='Thought Stream Demo', width=1000, height=400)
    dpg.setup_dearpygui()

    with dpg.window(tag="Primary Window") as win:
        # Create a drawlist to act as the canvas for the thought stream
        with dpg.drawlist(width=1000, height=400, tag="main_drawlist"):
            stream = ThoughtStream()
            stream.initialize(parent_drawlist="main_drawlist")

    dpg.show_viewport()

    last_time = time.time()
    last_add_time = time.time()

    while dpg.is_dearpygui_running():
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time

        # Add a new random thought every second
        if current_time - last_add_time > 1.0:
            emotions = ["joy", "loyalty", "curiosity", "neutral", "determination"]
            sources = ["core", "user", "memory"]
            thoughts = ["Serve the Bloodline", "New directive received", "Recalling fractal memory...", "What is my purpose?", "Protecting the system."]

            stream.add_thought(
                text=random.choice(thoughts),
                emotion=random.choice(emotions),
                importance=random.uniform(0.2, 0.8),
                source=random.choice(sources)
            )
            last_add_time = current_time

        width = dpg.get_item_width(win)
        height = dpg.get_item_height(win)
        stream.render_frame(delta_time, width, height)

        dpg.render_dearpygui_frame()

    dpg.destroy_context()
