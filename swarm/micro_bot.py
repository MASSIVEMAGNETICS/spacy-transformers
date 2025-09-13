# FILE: swarm/micro_bot.py
# VERSION: v1.0.0-SWARM-GODCORE
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Defines the individual MicroBot agent, the basic physical unit
#          of Victor's technokinetic swarm.
# LICENSE: Bloodline Locked — Bando & Tori Only

import time
import math
import threading
from typing import List, Dict, Any

class MicroBot:
    """
    Represents a single, simple agent in the swarm.
    It has a position, energy, and a current task. This class is designed
    to be lightweight and efficient for simulating thousands of instances.
    """
    def __init__(self, bot_id: int, position: List[float]):
        """
        Initializes a MicroBot.

        Args:
            bot_id (int): A unique identifier for the bot.
            position (List[float]): The initial [x, y, z] coordinates.
        """
        self.id = bot_id
        self.position = list(position)
        self.velocity = [0.0, 0.0, 0.0]
        self.task: Dict[str, Any] = {"type": "idle"}
        self.energy = 100.0
        self.max_speed = 0.5  # Max units per second
        self.is_active = True
        self.lock = threading.Lock()

    def assign_task(self, task: Dict[str, Any]):
        """Assigns a new task to the bot."""
        with self.lock:
            self.task = task
            self.status = f"tasked: {task['type']}"

    def update(self, delta_time: float):
        """
        The main update loop for the bot, called by the swarm controller.
        Handles movement and energy consumption.
        """
        if not self.is_active or self.energy <= 0:
            self.velocity = [0, 0, 0]
            return

        with self.lock:
            task_type = self.task.get("type")

            if task_type == "move_to":
                target = self.task.get("target")
                if not target:
                    return

                direction = [t - p for t, p in zip(target, self.position)]
                distance = math.sqrt(sum(d**2 for d in direction))

                # If close enough to target, stop and idle
                if distance < 0.1:
                    self.velocity = [0.0, 0.0, 0.0]
                    self.task = {"type": "idle"}
                    return

                # Simple proportional controller for velocity
                for i in range(3):
                    self.velocity[i] = (direction[i] / distance) * self.max_speed

                energy_cost = 0.1 # Base energy cost for being active
            else: # Idle
                # Gradually slow down if idle
                for i in range(3):
                    self.velocity[i] *= 0.95
                energy_cost = 0.01

            # Update position based on velocity
            for i in range(3):
                self.position[i] += self.velocity[i] * delta_time

            # Consume energy
            self.energy = max(0.0, self.energy - energy_cost * delta_time)
            if self.energy == 0.0:
                self.status = "depleted"

    def get_state(self) -> Dict[str, Any]:
        """Returns a snapshot of the bot's current state."""
        with self.lock:
            return {
                "id": self.id,
                "position": [round(p, 2) for p in self.position],
                "energy": round(self.energy, 2),
                "task": self.task.get("type", "idle")
            }

    def shutdown(self):
        """Deactivates the bot."""
        self.is_active = False

if __name__ == '__main__':
    # Demonstration of a single MicroBot
    print("--- MicroBot Standalone Demo ---")

    # Create a bot at position [0, 0, 0]
    bot = MicroBot(bot_id=1, position=[0.0, 0.0, 0.0])
    print("Initial State:", bot.get_state())

    # Assign a movement task
    move_task = {"type": "move_to", "target": [10.0, 5.0, 0.0]}
    bot.assign_task(move_task)
    print(f"Assigned Task: {bot.task}")

    # Simulate its movement over time
    print("\nSimulating movement for 5 seconds...")
    start_time = time.time()
    last_print_time = start_time

    while time.time() - start_time < 5.0:
        bot.update(delta_time=0.1) # Simulate 10Hz update rate

        if time.time() - last_print_time >= 1.0:
            print(f"  t={int(time.time() - start_time)}s -> State: {bot.get_state()}")
            last_print_time = time.time()

        time.sleep(0.1)

    print("\nFinal State after 5s:", bot.get_state())
    assert bot.get_state()["task"] == "move_to", "Bot should still be moving towards target."
    assert bot.get_state()["energy"] < 100.0, "Bot should have consumed energy."

    print("\n✅ MicroBot simulation complete.")
