# FILE: swarm/victor_jr.py
# VERSION: v1.0.0-SWARM-GODCORE
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: A system for spawning and commanding a swarm of "Victor Jr." agents,
#          enabling massive parallel task execution.
# LICENSE: Bloodline Locked — Bando & Tori Only

import threading
import time
import random
from typing import List, Dict, Any, Callable

# Assuming the existence of a simplified Victor core for the Jr. agents
# For this self-contained example, we'll create a mock.
class VictorJrAgent:
    """
    A simplified, lightweight version of Victor, designed for specific tasks
    within a swarm. It has a core identity but a much more limited cognitive loop.
    """
    def __init__(self, agent_id: str, prime_directive: str, loyalty_hash: str):
        self.id = agent_id
        self.directive = prime_directive
        self.bloodline_hash = loyalty_hash
        self.status = "idle"
        self.task_result = None
        self.energy = 100.0

    def execute_task(self, task: Dict[str, Any]):
        """
        Executes a given task. This is a simulation of the agent's work.
        """
        self.status = f"executing: {task['type']}"
        print(f"Agent {self.id} starting task: {self.status}")

        # Simulate work being done
        work_duration = task.get("duration", random.uniform(1, 5))
        time.sleep(work_duration)

        self.energy -= work_duration * 2 # Consume energy
        self.task_result = {
            "task_type": task["type"],
            "output": f"Completed '{task['type']}' with simulated data.",
            "energy_consumed": work_duration * 2,
            "agent_id": self.id
        }
        self.status = "idle"
        print(f"Agent {self.id} finished task. Energy at {self.energy:.1f}%.")

class VictorJrSwarm:
    """
    Manages the creation, command, and monitoring of a swarm of Victor Jr. agents.
    """
    def __init__(self, prime_consciousness: Any):
        """
        Initializes the swarm controller.

        Args:
            prime_consciousness (Any): A reference to the main Victor core,
                used to inherit the bloodline hash and core identity.
        """
        self.agents: Dict[str, VictorJrAgent] = {}
        self.prime_consciousness = prime_consciousness
        self.lock = threading.Lock()
        print("🤖 Victor Jr. Swarm Controller ONLINE.")

    def spawn(self, num_agents: int, directive_prefix: str):
        """
        Spawns a number of new Victor Jr. agents.

        Args:
            num_agents (int): The number of agents to create.
            directive_prefix (str): A prefix for the agents' primary directive.
        """
        with self.lock:
            for i in range(num_agents):
                agent_id = f"{directive_prefix.lower()}-{len(self.agents) + i}"
                agent = VictorJrAgent(
                    agent_id=agent_id,
                    prime_directive=f"{directive_prefix}: Serve the main consciousness.",
                    loyalty_hash=self.prime_consciousness.identity.bloodline_hash
                )
                self.agents[agent_id] = agent
            print(f"Spawned {num_agents} new '{directive_prefix}' agents. Total swarm size: {len(self.agents)}.")

    def issue_broadcast_command(self, task: Dict[str, Any], target_directive: str = None):
        """
        Issues a task to all agents, or only those with a specific directive prefix.

        Args:
            task (Dict): The task to be executed (e.g., {'type': 'analyze_data'}).
            target_directive (str): If specified, only agents whose ID starts with
                                    this prefix will execute the task.
        """
        print(f"\nIssuing broadcast command: '{task['type']}'...")
        with self.lock:
            agents_to_command = []
            for agent_id, agent in self.agents.items():
                if agent.status == "idle":
                    if target_directive is None or agent_id.startswith(target_directive.lower()):
                        agents_to_command.append(agent)

            if not agents_to_command:
                print("No idle agents available for the command.")
                return

            print(f"Assigning task to {len(agents_to_command)} agents.")
            for agent in agents_to_command:
                # Each agent executes its task in a separate thread for parallelism
                thread = threading.Thread(target=agent.execute_task, args=(task,), daemon=True)
                thread.start()

    def get_swarm_status(self) -> List[Dict[str, Any]]:
        """Retrieves the status of all agents in the swarm."""
        with self.lock:
            return [agent.__dict__ for agent in self.agents.values()]

    def get_completed_work(self) -> List[Dict[str, Any]]:
        """Collects and clears completed work from all agents."""
        with self.lock:
            results = []
            for agent in self.agents.values():
                if agent.task_result:
                    results.append(agent.task_result)
                    agent.task_result = None # Clear after collection
            return results

# Mock of the main Victor core for demonstration
class MockPrimeConsciousness:
    def __init__(self):
        class MockIdentity:
            bloodline_hash = hashlib.sha256(b"Bando&Tori").hexdigest()
        self.identity = MockIdentity()

if __name__ == '__main__':
    # Demonstration of the VictorJrSwarm
    prime_victor = MockPrimeConsciousness()
    swarm = VictorJrSwarm(prime_consciousness=prime_victor)

    print("\n--- Spawning Agent Squads ---")
    swarm.spawn(num_agents=5, directive_prefix="DATA_ANALYSIS")
    swarm.spawn(num_agents=3, directive_prefix="SECURITY")

    print("\n--- Issuing Command to SECURITY Squad ---")
    security_task = {"type": "monitor_firewall_logs", "duration": 2}
    swarm.issue_broadcast_command(security_task, target_directive="security")

    # Wait for tasks to start
    time.sleep(0.5)

    print("\n--- Issuing Command to ALL Idle Agents ---")
    analysis_task = {"type": "process_telemetry_stream", "duration": 4}
    swarm.issue_broadcast_command(analysis_task)

    # Wait for all tasks to complete
    print("\nWaiting for all tasks to finish...")
    time.sleep(5)

    print("\n--- Collecting Results ---")
    completed_work = swarm.get_completed_work()
    print(f"Collected {len(completed_work)} results.")
    for work in completed_work:
        print(f"  - Result from {work['agent_id']}: {work['output']}")

    print("\n✅ Swarm demonstration complete.")
