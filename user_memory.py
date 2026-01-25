#!/usr/bin/env python3
"""
BlackRoad User Memory System
Persistent memory and context management for AI agents.
Enables agents to remember users across sessions.
"""

import os
import json
import time
import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import uuid


class MemoryType(Enum):
    """Types of memories that can be stored"""
    CONVERSATION = "conversation"      # Chat history
    PREFERENCE = "preference"          # User preferences
    CONTEXT = "context"                # Session context
    FACT = "fact"                      # Learned facts about user
    INTERACTION = "interaction"        # Interaction patterns
    TASK_HISTORY = "task_history"      # Past task results


@dataclass
class MemoryEntry:
    """Single memory entry"""
    memory_id: str
    user_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    importance: float = 1.0  # 0.0 to 1.0, higher = more important
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "importance": self.importance,
            "access_count": self.access_count,
            "tags": self.tags,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryEntry":
        """Create from dictionary"""
        return cls(
            memory_id=data["memory_id"],
            user_id=data["user_id"],
            memory_type=MemoryType(data["memory_type"]),
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            importance=data.get("importance", 1.0),
            access_count=data.get("access_count", 0),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class UserProfile:
    """User profile with aggregated information"""
    user_id: str
    display_name: Optional[str] = None
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    total_interactions: int = 0
    preferences: Dict[str, Any] = field(default_factory=dict)
    facts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "total_interactions": self.total_interactions,
            "preferences": self.preferences,
            "facts": self.facts,
            "tags": self.tags,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UserProfile":
        return cls(
            user_id=data["user_id"],
            display_name=data.get("display_name"),
            first_seen=datetime.fromisoformat(data["first_seen"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            total_interactions=data.get("total_interactions", 0),
            preferences=data.get("preferences", {}),
            facts=data.get("facts", []),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )


class MemoryStore(ABC):
    """Abstract interface for memory storage backends"""

    @abstractmethod
    def store_memory(self, entry: MemoryEntry) -> bool:
        """Store a memory entry"""
        pass

    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID"""
        pass

    @abstractmethod
    def get_user_memories(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[MemoryEntry]:
        """Get all memories for a user, optionally filtered by type"""
        pass

    @abstractmethod
    def update_memory(self, memory_id: str, content: Dict[str, Any]) -> bool:
        """Update memory content"""
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory"""
        pass

    @abstractmethod
    def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user, returns count deleted"""
        pass

    @abstractmethod
    def search_memories(
        self,
        user_id: str,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search memories by content"""
        pass

    @abstractmethod
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        pass

    @abstractmethod
    def save_user_profile(self, profile: UserProfile) -> bool:
        """Save user profile"""
        pass

    @abstractmethod
    def cleanup_expired(self) -> int:
        """Remove expired memories, returns count removed"""
        pass

    @abstractmethod
    def delete_user_profile(self, user_id: str) -> bool:
        """Delete user profile (for GDPR compliance)"""
        pass


class InMemoryStore(MemoryStore):
    """
    In-memory storage backend for development and testing.
    Data is lost when the process ends.
    """

    def __init__(self):
        self._memories: Dict[str, MemoryEntry] = {}
        self._user_index: Dict[str, List[str]] = {}  # user_id -> [memory_ids]
        self._profiles: Dict[str, UserProfile] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger("InMemoryStore")

    def store_memory(self, entry: MemoryEntry) -> bool:
        with self._lock:
            self._memories[entry.memory_id] = entry
            if entry.user_id not in self._user_index:
                self._user_index[entry.user_id] = []
            if entry.memory_id not in self._user_index[entry.user_id]:
                self._user_index[entry.user_id].append(entry.memory_id)
            return True

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        with self._lock:
            entry = self._memories.get(memory_id)
            if entry:
                entry.access_count += 1
            return entry

    def get_user_memories(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[MemoryEntry]:
        with self._lock:
            memory_ids = self._user_index.get(user_id, [])
            memories = []
            for mid in memory_ids:
                entry = self._memories.get(mid)
                if entry:
                    if memory_type is None or entry.memory_type == memory_type:
                        # Skip expired
                        if entry.expires_at and entry.expires_at < datetime.now():
                            continue
                        memories.append(entry)

            # Sort by updated_at descending (most recent first)
            memories.sort(key=lambda x: x.updated_at, reverse=True)
            return memories[offset:offset + limit]

    def update_memory(self, memory_id: str, content: Dict[str, Any]) -> bool:
        with self._lock:
            if memory_id in self._memories:
                self._memories[memory_id].content = content
                self._memories[memory_id].updated_at = datetime.now()
                return True
            return False

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock:
            if memory_id in self._memories:
                entry = self._memories.pop(memory_id)
                if entry.user_id in self._user_index:
                    self._user_index[entry.user_id].remove(memory_id)
                return True
            return False

    def delete_user_memories(self, user_id: str) -> int:
        with self._lock:
            memory_ids = self._user_index.get(user_id, [])
            count = 0
            for mid in memory_ids[:]:
                if mid in self._memories:
                    del self._memories[mid]
                    count += 1
            self._user_index[user_id] = []
            return count

    def search_memories(
        self,
        user_id: str,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        with self._lock:
            query_lower = query.lower()
            results = []
            for entry in self.get_user_memories(user_id, memory_type, limit=1000):
                # Simple text search in content
                content_str = json.dumps(entry.content).lower()
                if query_lower in content_str:
                    results.append(entry)
                    if len(results) >= limit:
                        break
            return results

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        with self._lock:
            return self._profiles.get(user_id)

    def save_user_profile(self, profile: UserProfile) -> bool:
        with self._lock:
            self._profiles[profile.user_id] = profile
            return True

    def cleanup_expired(self) -> int:
        with self._lock:
            now = datetime.now()
            to_delete = []
            for mid, entry in self._memories.items():
                if entry.expires_at and entry.expires_at < now:
                    to_delete.append(mid)
            for mid in to_delete:
                self.delete_memory(mid)
            return len(to_delete)

    def delete_user_profile(self, user_id: str) -> bool:
        with self._lock:
            if user_id in self._profiles:
                del self._profiles[user_id]
                return True
            return False


class RedisMemoryStore(MemoryStore):
    """
    Redis-backed storage for production use.
    Provides persistence and fast access.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "blackroad:memory:"
    ):
        self.prefix = prefix
        self.logger = logging.getLogger("RedisMemoryStore")

        try:
            import redis
            self._redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True
            )
            self._redis.ping()
            self.logger.info(f"Connected to Redis at {host}:{port}")
        except ImportError:
            raise ImportError("redis package required. Install with: pip install redis")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    def _memory_key(self, memory_id: str) -> str:
        return f"{self.prefix}entry:{memory_id}"

    def _user_index_key(self, user_id: str) -> str:
        return f"{self.prefix}user:{user_id}:memories"

    def _profile_key(self, user_id: str) -> str:
        return f"{self.prefix}profile:{user_id}"

    def store_memory(self, entry: MemoryEntry) -> bool:
        try:
            key = self._memory_key(entry.memory_id)
            data = json.dumps(entry.to_dict())

            # Set with optional expiration
            if entry.expires_at:
                ttl = int((entry.expires_at - datetime.now()).total_seconds())
                if ttl > 0:
                    self._redis.setex(key, ttl, data)
                else:
                    return False  # Already expired
            else:
                self._redis.set(key, data)

            # Add to user index
            self._redis.sadd(self._user_index_key(entry.user_id), entry.memory_id)
            return True
        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            return False

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        try:
            data = self._redis.get(self._memory_key(memory_id))
            if data:
                entry = MemoryEntry.from_dict(json.loads(data))
                entry.access_count += 1
                # Update access count
                self._redis.set(self._memory_key(memory_id), json.dumps(entry.to_dict()))
                return entry
            return None
        except Exception as e:
            self.logger.error(f"Failed to get memory: {e}")
            return None

    def get_user_memories(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[MemoryEntry]:
        try:
            memory_ids = self._redis.smembers(self._user_index_key(user_id))
            memories = []

            for mid in memory_ids:
                entry = self.get_memory(mid)
                if entry:
                    if memory_type is None or entry.memory_type == memory_type:
                        memories.append(entry)

            # Sort by updated_at descending
            memories.sort(key=lambda x: x.updated_at, reverse=True)
            return memories[offset:offset + limit]
        except Exception as e:
            self.logger.error(f"Failed to get user memories: {e}")
            return []

    def update_memory(self, memory_id: str, content: Dict[str, Any]) -> bool:
        try:
            entry = self.get_memory(memory_id)
            if entry:
                entry.content = content
                entry.updated_at = datetime.now()
                return self.store_memory(entry)
            return False
        except Exception as e:
            self.logger.error(f"Failed to update memory: {e}")
            return False

    def delete_memory(self, memory_id: str) -> bool:
        try:
            entry = self.get_memory(memory_id)
            if entry:
                self._redis.delete(self._memory_key(memory_id))
                self._redis.srem(self._user_index_key(entry.user_id), memory_id)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete memory: {e}")
            return False

    def delete_user_memories(self, user_id: str) -> int:
        try:
            memory_ids = self._redis.smembers(self._user_index_key(user_id))
            count = 0
            for mid in memory_ids:
                if self._redis.delete(self._memory_key(mid)):
                    count += 1
            self._redis.delete(self._user_index_key(user_id))
            return count
        except Exception as e:
            self.logger.error(f"Failed to delete user memories: {e}")
            return 0

    def search_memories(
        self,
        user_id: str,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        # Simple implementation - for production, use Redis Search or external search
        memories = self.get_user_memories(user_id, memory_type, limit=1000)
        query_lower = query.lower()
        results = []
        for entry in memories:
            content_str = json.dumps(entry.content).lower()
            if query_lower in content_str:
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        try:
            data = self._redis.get(self._profile_key(user_id))
            if data:
                return UserProfile.from_dict(json.loads(data))
            return None
        except Exception as e:
            self.logger.error(f"Failed to get profile: {e}")
            return None

    def save_user_profile(self, profile: UserProfile) -> bool:
        try:
            self._redis.set(
                self._profile_key(profile.user_id),
                json.dumps(profile.to_dict())
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to save profile: {e}")
            return False

    def cleanup_expired(self) -> int:
        # Redis handles expiration automatically via TTL
        return 0

    def delete_user_profile(self, user_id: str) -> bool:
        try:
            return bool(self._redis.delete(self._profile_key(user_id)))
        except Exception as e:
            self.logger.error(f"Failed to delete profile: {e}")
            return False


class UserMemory:
    """
    High-level user memory management.
    Provides convenient methods for storing and retrieving user context.
    """

    def __init__(self, store: Optional[MemoryStore] = None):
        """
        Initialize UserMemory with a storage backend.
        Defaults to InMemoryStore if none provided.
        """
        self.store = store or InMemoryStore()
        self.logger = logging.getLogger("UserMemory")

    @classmethod
    def create_with_redis(
        cls,
        host: str = None,
        port: int = None,
        password: str = None
    ) -> "UserMemory":
        """Factory method to create UserMemory with Redis backend"""
        host = host or os.getenv("REDIS_HOST", "localhost")
        port = port or int(os.getenv("REDIS_PORT", "6379"))
        password = password or os.getenv("REDIS_PASSWORD")

        store = RedisMemoryStore(host=host, port=port, password=password)
        return cls(store=store)

    def _generate_memory_id(self) -> str:
        """Generate unique memory ID"""
        return f"mem_{uuid.uuid4().hex[:12]}"

    def _get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get existing profile or create new one"""
        profile = self.store.get_user_profile(user_id)
        if not profile:
            profile = UserProfile(user_id=user_id)
            self.store.save_user_profile(profile)
        return profile

    def remember(
        self,
        user_id: str,
        content: Dict[str, Any],
        memory_type: MemoryType = MemoryType.CONTEXT,
        importance: float = 1.0,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Store a memory for a user.
        Returns memory_id.
        """
        now = datetime.now()
        entry = MemoryEntry(
            memory_id=self._generate_memory_id(),
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds) if ttl_seconds else None,
            importance=importance,
            tags=tags or []
        )

        self.store.store_memory(entry)

        # Update profile
        profile = self._get_or_create_profile(user_id)
        profile.last_seen = now
        profile.total_interactions += 1
        self.store.save_user_profile(profile)

        self.logger.debug(f"Stored memory {entry.memory_id} for user {user_id}")
        return entry.memory_id

    def remember_conversation(
        self,
        user_id: str,
        role: str,
        message: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Store a conversation message"""
        return self.remember(
            user_id=user_id,
            content={
                "role": role,
                "message": message,
                "metadata": metadata or {}
            },
            memory_type=MemoryType.CONVERSATION,
            tags=["conversation", role]
        )

    def remember_preference(
        self,
        user_id: str,
        key: str,
        value: Any,
        category: str = "general"
    ) -> str:
        """Store a user preference"""
        # Also update profile preferences
        profile = self._get_or_create_profile(user_id)
        profile.preferences[key] = value
        self.store.save_user_profile(profile)

        return self.remember(
            user_id=user_id,
            content={
                "key": key,
                "value": value,
                "category": category
            },
            memory_type=MemoryType.PREFERENCE,
            importance=0.8,
            tags=["preference", category]
        )

    def remember_fact(
        self,
        user_id: str,
        fact: str,
        confidence: float = 1.0,
        source: str = "conversation"
    ) -> str:
        """Store a learned fact about the user"""
        # Also update profile facts
        profile = self._get_or_create_profile(user_id)
        if fact not in profile.facts:
            profile.facts.append(fact)
        self.store.save_user_profile(profile)

        return self.remember(
            user_id=user_id,
            content={
                "fact": fact,
                "confidence": confidence,
                "source": source
            },
            memory_type=MemoryType.FACT,
            importance=confidence,
            tags=["fact", source]
        )

    def recall(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 50
    ) -> List[MemoryEntry]:
        """Recall memories for a user"""
        return self.store.get_user_memories(user_id, memory_type, limit)

    def recall_conversations(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[Dict]:
        """Get recent conversation history"""
        memories = self.recall(user_id, MemoryType.CONVERSATION, limit)
        return [
            {
                "role": m.content.get("role"),
                "message": m.content.get("message"),
                "timestamp": m.created_at.isoformat()
            }
            for m in memories
        ]

    def recall_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get all user preferences"""
        profile = self.store.get_user_profile(user_id)
        if profile:
            return profile.preferences
        return {}

    def recall_facts(self, user_id: str) -> List[str]:
        """Get all learned facts about user"""
        profile = self.store.get_user_profile(user_id)
        if profile:
            return profile.facts
        return []

    def search(
        self,
        user_id: str,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search user memories"""
        return self.store.search_memories(user_id, query, memory_type, limit)

    def forget(self, memory_id: str) -> bool:
        """Delete a specific memory"""
        return self.store.delete_memory(memory_id)

    def forget_user(self, user_id: str) -> int:
        """Delete all memories for a user (GDPR compliance)"""
        count = self.store.delete_user_memories(user_id)
        # Also delete profile
        self.store.delete_user_profile(user_id)
        self.logger.info(f"Deleted {count} memories for user {user_id}")
        return count

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        return self.store.get_user_profile(user_id)

    def get_context_summary(self, user_id: str) -> Dict:
        """
        Get a summary of user context for injection into prompts.
        This is what agents use to "remember" users.
        """
        profile = self.store.get_user_profile(user_id)
        if not profile:
            return {"known": False, "user_id": user_id}

        recent_conversations = self.recall_conversations(user_id, limit=5)

        return {
            "known": True,
            "user_id": user_id,
            "display_name": profile.display_name,
            "first_seen": profile.first_seen.isoformat(),
            "last_seen": profile.last_seen.isoformat(),
            "total_interactions": profile.total_interactions,
            "preferences": profile.preferences,
            "facts": profile.facts,
            "recent_conversations": recent_conversations,
            "tags": profile.tags
        }

    def update_profile(
        self,
        user_id: str,
        display_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Update user profile information"""
        profile = self._get_or_create_profile(user_id)

        if display_name is not None:
            profile.display_name = display_name
        if tags is not None:
            profile.tags = tags
        if metadata is not None:
            profile.metadata.update(metadata)

        profile.last_seen = datetime.now()
        return self.store.save_user_profile(profile)


class MemoryEnabledMixin:
    """
    Mixin to add memory capabilities to BaseAgent.
    Add this to your agent class to enable user memory.
    """

    def init_memory(self, memory: Optional[UserMemory] = None):
        """Initialize memory system. Call this in agent __init__"""
        self.memory = memory or UserMemory()
        self._memory_enabled = True

    def process_task_with_memory(self, task: Dict) -> Dict:
        """
        Process task with memory context.
        Wraps process_task to inject user context.
        """
        user_id = task.get("user_id")

        if user_id and hasattr(self, "memory"):
            # Get user context
            context = self.memory.get_context_summary(user_id)
            task["_user_context"] = context

            # Record interaction
            self.memory.remember(
                user_id=user_id,
                content={
                    "task_type": task.get("task_type"),
                    "agent_id": getattr(self, "agent_id", "unknown"),
                    "timestamp": datetime.now().isoformat()
                },
                memory_type=MemoryType.INTERACTION
            )

        # Call original process_task
        result = self.process_task(task)

        # Store task result in memory
        if user_id and hasattr(self, "memory"):
            self.memory.remember(
                user_id=user_id,
                content={
                    "task_id": task.get("task_id"),
                    "success": result.get("success"),
                    "agent_id": getattr(self, "agent_id", "unknown")
                },
                memory_type=MemoryType.TASK_HISTORY
            )

        return result


# Utility functions for easy integration
def get_user_hash(identifier: str) -> str:
    """Generate consistent user ID from any identifier"""
    return hashlib.sha256(identifier.encode()).hexdigest()[:16]


def create_memory_system(backend: str = "memory") -> UserMemory:
    """
    Factory function to create memory system.
    backend: "memory" for in-memory, "redis" for Redis
    """
    if backend == "redis":
        return UserMemory.create_with_redis()
    return UserMemory()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("BlackRoad User Memory System - Demo")
    print("=" * 60)

    # Create memory system
    memory = UserMemory()

    # Simulate a user
    user_id = "user_12345"

    # Store some memories
    print("\n1. Storing memories...")

    memory.remember_conversation(user_id, "user", "Hello, I'm interested in quantum physics")
    memory.remember_conversation(user_id, "assistant", "Great! What aspect interests you most?")
    memory.remember_conversation(user_id, "user", "I want to learn about wave functions")

    memory.remember_preference(user_id, "topic", "quantum_physics", "learning")
    memory.remember_preference(user_id, "difficulty", "intermediate", "learning")

    memory.remember_fact(user_id, "User is learning quantum physics")
    memory.remember_fact(user_id, "User prefers intermediate difficulty")

    # Update profile
    memory.update_profile(user_id, display_name="Physics Student")

    # Recall memories
    print("\n2. Recalling context summary...")
    context = memory.get_context_summary(user_id)
    print(json.dumps(context, indent=2, default=str))

    # Search memories
    print("\n3. Searching memories for 'quantum'...")
    results = memory.search(user_id, "quantum")
    for r in results:
        print(f"  - {r.memory_type.value}: {r.content}")

    # Check if user is "remembered"
    print("\n4. Do I remember this user?")
    if context["known"]:
        print(f"  Yes! Hello {context['display_name']}, good to see you again!")
        print(f"  I remember you're interested in: {context['facts']}")
    else:
        print("  No, this is a new user.")

    print("\n" + "=" * 60)
    print("Memory system ready for integration!")
    print("=" * 60)
