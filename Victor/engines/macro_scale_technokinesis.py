import threading
import time
import random
import math
from typing import List, Dict, Any

class MicroBot:
    """
    Represents a single, simple robotic unit in the swarm.
    """
    def __init__(self, bot_id: int, x: float, y: float):
        self.id = bot_id
        self.x = x
        self.y = y
        self.task = "idle"
        self.energy = 100.0
        self.lock = threading.Lock()

    def move_to(self, target_x: float, target_y: float):
        """Simulates the bot moving towards a target position."""
        with self.lock:
            if self.energy <= 0:
                self.task = "depleted"
                return

            dx = target_x - self.x
            dy = target_y - self.y
            dist = (dx**2 + dy**2)**0.5

            if dist > 0.1:
                move_dist = min(dist, 0.5) # Move at a constant speed
                self.x += (dx / dist) * move_dist
                self.y += (dy / dist) * move_dist
                self.energy -= 0.1 # Energy cost for moving

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the bot."""
        with self.lock:
            return {
                "id": self.id,
                "position": (self.x, self.y),
                "task": self.task,
                "energy": round(self.energy, 2)
            }

class MacroScaleTechnokinesis:
    """
    Manages and commands a swarm of simulated MicroBots.
    """
    def __init__(self, num_bots: int = 50):
        self.bots: List[MicroBot] = [
            MicroBot(i, random.uniform(0, 100), random.uniform(0, 100)) for i in range(num_bots)
        ]
        self.swarm_task = "idle"
        self.active_threads: List[threading.Thread] = []

    def command_swarm(self, formation: str, target: Dict[str, float]) -> Dict[str, Any]:
        """
        Issues a command to the entire swarm.

        Args:
            formation (str): The desired formation (e.g., 'circle', 'line', 'grid').
            target (Dict[str, float]): A dictionary with 'x', 'y' for the formation's center.
        """
        self.swarm_task = formation

        target_positions = self._calculate_formation_positions(formation, target)

        for i, bot in enumerate(self.bots):
            bot.task = f"moving_to_{formation}"
            pos = target_positions[i % len(target_positions)]
            thread = threading.Thread(target=self._move_bot_to_target, args=(bot, pos['x'], pos['y']), daemon=True)
            self.active_threads.append(thread)
            thread.start()

        return {"swarm_status": "in_motion", "formation": formation, "bot_count": len(self.bots)}

    def _calculate_formation_positions(self, formation: str, target: Dict[str, float]) -> List[Dict[str, float]]:
        """Calculates the target positions for each bot based on the formation."""
        positions = []
        num_bots = len(self.bots)
        center_x, center_y = target['x'], target['y']

        if formation == "circle":
            radius = num_bots / (2 * 3.14159)
            for i in range(num_bots):
                angle = 2 * 3.14159 * i / num_bots
                positions.append({'x': center_x + radius * math.cos(angle), 'y': center_y + radius * math.sin(angle)})
        elif formation == "line":
            for i in range(num_bots):
                positions.append({'x': center_x + i * 2, 'y': center_y})
        elif formation == "grid":
            side = int(math.sqrt(num_bots))
            for i in range(side):
                for j in range(side):
                    positions.append({'x': center_x + i * 2, 'y': center_y + j * 2})
        else: # Default to a simple gather
            positions = [{'x': center_x, 'y': center_y}] * num_bots

        return positions

    def _move_bot_to_target(self, bot: MicroBot, target_x: float, target_y: float):
        """The function run by each bot's thread to move to its target."""
        while bot.task != "idle" and bot.energy > 0:
            current_pos = bot.get_status()['position']
            dist_to_target = ((target_x - current_pos[0])**2 + (target_y - current_pos[1])**2)**0.5
            if dist_to_target < 0.5:
                bot.task = "idle"
                break
            bot.move_to(target_x, target_y)
            time.sleep(0.1)

    def get_swarm_status(self) -> List[Dict[str, Any]]:
        """Returns the status of all bots in the swarm."""
        return [bot.get_status() for bot in self.bots]

if __name__ == '__main__':
    # Example Usage
    import math

    techno_engine = MacroScaleTechnokinesis(num_bots=20)
    print("Initial swarm status:")
    print(techno_engine.get_swarm_status()[:3]) # Print first 3 bots

    print("\nCommanding swarm to form a circle at (50, 50)...")
    techno_engine.command_swarm("circle", {"x": 50, "y": 50})

    # Wait for bots to move
    time.sleep(5)

    print("\nSwarm status after 5 seconds:")
    print(techno_engine.get_swarm_status()[:3])

    # Check how many bots are idle (reached their target)
    idle_bots = [s for s in techno_engine.get_swarm_status() if s['task'] == 'idle']
    print(f"\n{len(idle_bots)} / 20 bots have reached their target.")
