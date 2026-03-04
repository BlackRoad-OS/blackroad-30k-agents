#!/usr/bin/env python3
"""
BlackRoad Memory-Enabled Agent
Agent with built-in user memory capabilities.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

from base_agent import BaseAgent, AgentType, AgentStatus
from user_memory import (
    UserMemory,
    MemoryEnabledMixin,
    MemoryType,
    create_memory_system
)


class MemoryEnabledAgent(MemoryEnabledMixin, BaseAgent):
    """
    Agent with user memory capabilities.
    Remembers users across sessions and provides personalized interactions.
    """

    def __init__(
        self,
        agent_type: AgentType,
        capabilities: List[str],
        agent_id: Optional[str] = None,
        port: int = 8080,
        memory_backend: str = "memory"
    ):
        # Initialize BaseAgent
        super().__init__(
            agent_type=agent_type,
            capabilities=capabilities,
            agent_id=agent_id,
            port=port
        )

        # Initialize memory system
        self.init_memory(create_memory_system(memory_backend))
        self.capabilities.append("user_memory")

        self.logger.info(f"Memory system initialized (backend: {memory_backend})")

    def process_task(self, task: Dict) -> Dict:
        """
        Process task with memory context injection.
        Automatically injects user context if user_id is provided.
        """
        user_id = task.get("user_id")

        # Inject user context if user is known
        if user_id:
            context = self.memory.get_context_summary(user_id)
            task["_user_context"] = context

            # Log if returning user
            if context.get("known"):
                self.logger.info(
                    f"Recognized user: {context.get('display_name', user_id)} "
                    f"(interactions: {context.get('total_interactions', 0)})"
                )

                # Record this interaction
                self.memory.remember(
                    user_id=user_id,
                    content={
                        "task_type": task.get("task_type", "unknown"),
                        "agent_id": self.agent_id,
                        "agent_type": self.agent_type.value
                    },
                    memory_type=MemoryType.INTERACTION
                )

        # Process with parent class
        result = super().process_task(task)

        # Store task result in history
        if user_id:
            self.memory.remember(
                user_id=user_id,
                content={
                    "task_id": task.get("task_id"),
                    "task_type": task.get("task_type"),
                    "success": result.get("success"),
                    "agent_id": self.agent_id,
                    "processing_time": result.get("processing_time_s")
                },
                memory_type=MemoryType.TASK_HISTORY,
                ttl_seconds=86400 * 30  # Keep for 30 days
            )

        return result

    def start_http_server(self):
        """Start HTTP server with memory endpoints"""
        agent = self

        class MemoryAgentHTTPHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def _send_json(self, data: Dict, status: int = 200):
                self.send_response(status)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def _read_json(self) -> Dict:
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length:
                    return json.loads(self.rfile.read(content_length).decode())
                return {}

            def do_GET(self):
                # Standard health endpoints
                if self.path == "/health":
                    self._send_json(agent.health_check())

                elif self.path == "/ready":
                    health = agent.readiness_check()
                    status_code = 200 if health["ready"] else 503
                    self._send_json(health, status_code)

                elif self.path == "/metrics":
                    self._send_json(agent.get_metrics())

                # Memory endpoints
                elif self.path.startswith("/memory/user/"):
                    # GET /memory/user/{user_id}
                    user_id = self.path.split("/")[-1]
                    context = agent.memory.get_context_summary(user_id)
                    self._send_json(context)

                elif self.path.startswith("/memory/profile/"):
                    # GET /memory/profile/{user_id}
                    user_id = self.path.split("/")[-1]
                    profile = agent.memory.get_profile(user_id)
                    if profile:
                        self._send_json(profile.to_dict())
                    else:
                        self._send_json({"error": "User not found"}, 404)

                elif self.path.startswith("/memory/conversations/"):
                    # GET /memory/conversations/{user_id}
                    user_id = self.path.split("/")[-1]
                    conversations = agent.memory.recall_conversations(user_id)
                    self._send_json({"conversations": conversations})

                elif self.path.startswith("/memory/preferences/"):
                    # GET /memory/preferences/{user_id}
                    user_id = self.path.split("/")[-1]
                    preferences = agent.memory.recall_preferences(user_id)
                    self._send_json({"preferences": preferences})

                elif self.path.startswith("/memory/facts/"):
                    # GET /memory/facts/{user_id}
                    user_id = self.path.split("/")[-1]
                    facts = agent.memory.recall_facts(user_id)
                    self._send_json({"facts": facts})

                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == "/task":
                    task_data = self._read_json()
                    result = agent.process_task(task_data)
                    self._send_json(result)

                # Memory write endpoints
                elif self.path == "/memory/conversation":
                    # POST /memory/conversation
                    # Body: {user_id, role, message}
                    data = self._read_json()
                    memory_id = agent.memory.remember_conversation(
                        user_id=data["user_id"],
                        role=data["role"],
                        message=data["message"],
                        metadata=data.get("metadata")
                    )
                    self._send_json({"memory_id": memory_id})

                elif self.path == "/memory/preference":
                    # POST /memory/preference
                    # Body: {user_id, key, value, category?}
                    data = self._read_json()
                    memory_id = agent.memory.remember_preference(
                        user_id=data["user_id"],
                        key=data["key"],
                        value=data["value"],
                        category=data.get("category", "general")
                    )
                    self._send_json({"memory_id": memory_id})

                elif self.path == "/memory/fact":
                    # POST /memory/fact
                    # Body: {user_id, fact, confidence?}
                    data = self._read_json()
                    memory_id = agent.memory.remember_fact(
                        user_id=data["user_id"],
                        fact=data["fact"],
                        confidence=data.get("confidence", 1.0)
                    )
                    self._send_json({"memory_id": memory_id})

                elif self.path == "/memory/profile":
                    # POST /memory/profile
                    # Body: {user_id, display_name?, tags?, metadata?}
                    data = self._read_json()
                    success = agent.memory.update_profile(
                        user_id=data["user_id"],
                        display_name=data.get("display_name"),
                        tags=data.get("tags"),
                        metadata=data.get("metadata")
                    )
                    self._send_json({"success": success})

                elif self.path == "/memory/search":
                    # POST /memory/search
                    # Body: {user_id, query, memory_type?, limit?}
                    data = self._read_json()
                    memory_type = None
                    if data.get("memory_type"):
                        memory_type = MemoryType(data["memory_type"])

                    results = agent.memory.search(
                        user_id=data["user_id"],
                        query=data["query"],
                        memory_type=memory_type,
                        limit=data.get("limit", 10)
                    )
                    self._send_json({
                        "results": [r.to_dict() for r in results]
                    })

                else:
                    self.send_response(404)
                    self.end_headers()

            def do_DELETE(self):
                if self.path.startswith("/memory/user/"):
                    # DELETE /memory/user/{user_id} - GDPR delete all
                    user_id = self.path.split("/")[-1]
                    count = agent.memory.forget_user(user_id)
                    self._send_json({
                        "deleted": True,
                        "memories_removed": count
                    })

                elif self.path.startswith("/memory/"):
                    # DELETE /memory/{memory_id}
                    memory_id = self.path.split("/")[-1]
                    success = agent.memory.forget(memory_id)
                    self._send_json({"deleted": success})

                else:
                    self.send_response(404)
                    self.end_headers()

        server = HTTPServer(("0.0.0.0", self.port), MemoryAgentHTTPHandler)
        self.logger.info(f"Memory-enabled HTTP server started on port {self.port}")
        self.logger.info("Memory endpoints available:")
        self.logger.info("  GET  /memory/user/{user_id} - Get user context")
        self.logger.info("  GET  /memory/profile/{user_id} - Get user profile")
        self.logger.info("  POST /memory/conversation - Store conversation")
        self.logger.info("  POST /memory/preference - Store preference")
        self.logger.info("  POST /memory/fact - Store fact")
        self.logger.info("  POST /memory/search - Search memories")
        self.logger.info("  DELETE /memory/user/{user_id} - Delete all user data")
        server.serve_forever()


# Example: Memory-enabled Quantum Mechanics Agent
class MemoryQuantumAgent(MemoryEnabledAgent):
    """Quantum Mechanics agent with user memory"""

    def __init__(self, agent_id: Optional[str] = None, memory_backend: str = "memory"):
        super().__init__(
            agent_type=AgentType.QUANTUM_MECHANICS,
            capabilities=[
                "hydrogen_energy",
                "wave_function",
                "uncertainty",
                "harmonic_oscillator",
                "personalized_explanations"
            ],
            agent_id=agent_id,
            memory_backend=memory_backend
        )

    def execute_task(self, task: Dict) -> Dict:
        """Execute quantum mechanics task with personalization"""
        calc_type = task.get("calculation_type")
        user_context = task.get("_user_context", {})

        # Personalize response based on user history
        difficulty = "intermediate"
        if user_context.get("known"):
            prefs = user_context.get("preferences", {})
            difficulty = prefs.get("difficulty", "intermediate")

        if calc_type == "hydrogen_energy":
            n = task.get("n", 1)
            energy = -13.6 / (n ** 2)

            result = {"energy_eV": energy, "n": n}

            # Add explanation based on user level
            if difficulty == "beginner":
                result["explanation"] = (
                    f"The hydrogen atom at energy level n={n} has an energy of {energy:.2f} eV. "
                    "This is like the electron being on a specific step of a ladder."
                )
            elif difficulty == "advanced":
                result["explanation"] = (
                    f"E_n = -13.6/n^2 eV yields E_{n} = {energy:.4f} eV for the "
                    f"principal quantum number n={n}, derived from the Bohr model."
                )

            return result

        elif calc_type == "uncertainty":
            delta_x = task.get("delta_x")
            h_bar = 1.054571817e-34
            delta_p_min = h_bar / (2 * delta_x)
            return {"delta_p_min": delta_p_min}

        else:
            raise ValueError(f"Unknown calculation type: {calc_type}")


# Entry point
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configuration from environment
    agent_type_str = os.getenv("AGENT_TYPE", "quantum_mechanics")
    agent_id = os.getenv("AGENT_ID")
    port = int(os.getenv("PORT", "8080"))
    memory_backend = os.getenv("MEMORY_BACKEND", "memory")

    print("=" * 60)
    print("BlackRoad Memory-Enabled Agent")
    print("=" * 60)
    print(f"Agent Type: {agent_type_str}")
    print(f"Memory Backend: {memory_backend}")
    print(f"Port: {port}")
    print("=" * 60)

    # Create appropriate agent
    if agent_type_str == "quantum_mechanics":
        agent = MemoryQuantumAgent(
            agent_id=agent_id,
            memory_backend=memory_backend
        )
    else:
        # Generic memory-enabled agent
        try:
            agent_type = AgentType(agent_type_str)
        except ValueError:
            print(f"Unknown agent type: {agent_type_str}")
            sys.exit(1)

        agent = MemoryEnabledAgent(
            agent_type=agent_type,
            capabilities=["general", "user_memory"],
            agent_id=agent_id,
            memory_backend=memory_backend
        )

    # Run agent
    agent.run()
