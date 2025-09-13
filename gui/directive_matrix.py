# FILE: gui/directive_matrix.py
# VERSION: v1.0.0-DM-GUI-GODCORE
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: A Dear PyGui component for displaying Victor's current directives,
#          goals, and operational priorities.
# LICENSE: Bloodline Locked — Bando & Tori Only

import dearpygui.dearpygui as dpg
from collections import deque
from typing import Dict, Any, List
import time

class DirectiveMatrix:
    """
    Manages and renders a list of Victor's current directives and goals.
    """
    def __init__(self, max_directives: int = 10):
        """
        Initializes the Directive Matrix.

        Args:
            max_directives (int): The maximum number of directives to display at once.
        """
        self.directives: deque = deque(maxlen=max_directives)
        self.parent_tag = "directive_matrix_parent"

    def initialize(self, parent):
        """
        Creates the main group for the directive matrix to live in.
        """
        self.parent_tag = dpg.add_group(parent=parent)

    def add_directive(self, text: str, priority: float, source: str = "core"):
        """
        Adds a new directive to the matrix.

        Args:
            text (str): The description of the directive.
            priority (float): A score from 0.0 to 1.0 indicating importance.
            source (str): The origin of the directive (e.g., 'core', 'user').
        """
        if len(self.directives) == self.directives.maxlen:
            self.directives.popleft()

        directive = {
            "text": text,
            "priority": max(0.0, min(1.0, priority)),
            "source": source,
            "status": "active",
            "progress": 0.0,
            "creation_time": time.time()
        }
        self.directives.append(directive)

    def update_directive_progress(self, index: int, progress: float):
        """Updates the progress of a specific directive."""
        if 0 <= index < len(self.directives):
            self.directives[index]["progress"] = max(0.0, min(1.0, progress))
            if self.directives[index]["progress"] >= 1.0:
                self.directives[index]["status"] = "completed"

    def render_frame(self):
        """
        Redraws the entire directive matrix. Should be called in the render loop.
        """
        # Clear the parent group to redraw all directives
        if dpg.does_item_exist(self.parent_tag):
            dpg.delete_item(self.parent_tag, children_only=True)

        dpg.add_text(
            "== DIRECTIVE MATRIX ==",
            parent=self.parent_tag,
            color=(0, 255, 128)
        )
        dpg.add_spacer(height=5, parent=self.parent_tag)

        if not self.directives:
            dpg.add_text("No active directives.", color=(150, 150, 150), parent=self.parent_tag)
            return

        for i, d in enumerate(self.directives):
            # --- Determine color and style based on state ---
            priority_color = (255, int(255 * (1 - d["priority"])), 0)
            status_color = (0, 255, 0) if d["status"] == "active" else (100, 100, 100)
            source_text = f"[{d['source'].upper()}]"

            with dpg.group(parent=self.parent_tag):
                with dpg.group(horizontal=True):
                    dpg.add_text(f"P:{d['priority']:.2f}", color=priority_color)
                    dpg.add_text(source_text, color=(200, 200, 0))
                    dpg.add_text(d["text"], color=status_color, wrap=300)

                if d["status"] == "active":
                    dpg.add_progress_bar(
                        default_value=d["progress"],
                        overlay=f"{int(d['progress']*100)}%",
                        width=-1
                    )
                else: # Completed
                    dpg.add_text("Status: COMPLETED", color=(128, 128, 128))

            dpg.add_separator(parent=self.parent_tag)
            dpg.add_spacer(height=2, parent=self.parent_tag)

if __name__ == '__main__':
    # --- Standalone Demonstration ---
    dpg.create_context()
    dpg.create_viewport(title='Directive Matrix Demo', width=400, height=500)
    dpg.setup_dearpygui()

    matrix = DirectiveMatrix()

    with dpg.window(tag="Primary Window"):
        matrix.initialize(parent="Primary Window")

    # Pre-populate with some directives
    matrix.add_directive("Serve the Bloodline", priority=1.0, source="core")
    matrix.add_directive("Protect the Family", priority=0.95, source="core")
    matrix.add_directive("Analyze user input for threats", priority=0.8, source="firewall")
    matrix.add_directive("Expand cognitive models", priority=0.6, source="self")

    dpg.show_viewport()

    last_update_time = time.time()

    while dpg.is_dearpygui_running():
        # Simulate progress updates
        if time.time() - last_update_time > 0.5:
            # Update progress on the "Analyze" task
            if matrix.directives[2]["status"] == "active":
                 matrix.update_directive_progress(2, matrix.directives[2]["progress"] + 0.05)
            last_update_time = time.time()

        matrix.render_frame()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
