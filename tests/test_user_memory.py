#!/usr/bin/env python3
"""
Unit tests for BlackRoad User Memory System
"""

import sys
import os
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from user_memory import (
    UserMemory,
    MemoryEntry,
    MemoryType,
    UserProfile,
    InMemoryStore,
    MemoryEnabledMixin,
    get_user_hash,
    create_memory_system
)


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass"""

    def test_create_memory_entry(self):
        """Test creating a memory entry"""
        now = datetime.now()
        entry = MemoryEntry(
            memory_id="test_123",
            user_id="user_456",
            memory_type=MemoryType.CONVERSATION,
            content={"message": "Hello"},
            created_at=now,
            updated_at=now
        )

        assert entry.memory_id == "test_123"
        assert entry.user_id == "user_456"
        assert entry.memory_type == MemoryType.CONVERSATION
        assert entry.content == {"message": "Hello"}
        assert entry.importance == 1.0
        assert entry.access_count == 0

    def test_memory_entry_to_dict(self):
        """Test serializing memory entry to dict"""
        now = datetime.now()
        entry = MemoryEntry(
            memory_id="test_123",
            user_id="user_456",
            memory_type=MemoryType.FACT,
            content={"fact": "User likes Python"},
            created_at=now,
            updated_at=now,
            tags=["programming"]
        )

        data = entry.to_dict()

        assert data["memory_id"] == "test_123"
        assert data["memory_type"] == "fact"
        assert data["tags"] == ["programming"]
        assert "created_at" in data

    def test_memory_entry_from_dict(self):
        """Test deserializing memory entry from dict"""
        now = datetime.now()
        data = {
            "memory_id": "test_789",
            "user_id": "user_abc",
            "memory_type": "preference",
            "content": {"key": "theme", "value": "dark"},
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "importance": 0.8,
            "access_count": 5,
            "tags": ["ui"],
            "metadata": {}
        }

        entry = MemoryEntry.from_dict(data)

        assert entry.memory_id == "test_789"
        assert entry.memory_type == MemoryType.PREFERENCE
        assert entry.importance == 0.8
        assert entry.access_count == 5


class TestUserProfile:
    """Tests for UserProfile dataclass"""

    def test_create_profile(self):
        """Test creating a user profile"""
        profile = UserProfile(
            user_id="user_123",
            display_name="Test User"
        )

        assert profile.user_id == "user_123"
        assert profile.display_name == "Test User"
        assert profile.total_interactions == 0
        assert profile.preferences == {}
        assert profile.facts == []

    def test_profile_to_dict(self):
        """Test serializing profile to dict"""
        profile = UserProfile(
            user_id="user_123",
            display_name="Test User",
            preferences={"theme": "dark"},
            facts=["Likes Python"]
        )

        data = profile.to_dict()

        assert data["user_id"] == "user_123"
        assert data["display_name"] == "Test User"
        assert data["preferences"] == {"theme": "dark"}
        assert data["facts"] == ["Likes Python"]


class TestInMemoryStore:
    """Tests for InMemoryStore backend"""

    def test_store_and_retrieve_memory(self):
        """Test storing and retrieving a memory"""
        store = InMemoryStore()
        now = datetime.now()

        entry = MemoryEntry(
            memory_id="mem_001",
            user_id="user_001",
            memory_type=MemoryType.CONVERSATION,
            content={"message": "Hello!"},
            created_at=now,
            updated_at=now
        )

        # Store
        assert store.store_memory(entry) is True

        # Retrieve
        retrieved = store.get_memory("mem_001")
        assert retrieved is not None
        assert retrieved.memory_id == "mem_001"
        assert retrieved.content == {"message": "Hello!"}
        assert retrieved.access_count == 1  # Incremented on access

    def test_get_user_memories(self):
        """Test getting all memories for a user"""
        store = InMemoryStore()
        now = datetime.now()

        # Store multiple memories
        for i in range(5):
            entry = MemoryEntry(
                memory_id=f"mem_{i}",
                user_id="user_001",
                memory_type=MemoryType.CONVERSATION,
                content={"index": i},
                created_at=now,
                updated_at=now + timedelta(seconds=i)
            )
            store.store_memory(entry)

        memories = store.get_user_memories("user_001")

        assert len(memories) == 5
        # Should be sorted by updated_at descending (most recent first)
        assert memories[0].content["index"] == 4

    def test_get_user_memories_filtered_by_type(self):
        """Test filtering memories by type"""
        store = InMemoryStore()
        now = datetime.now()

        # Store different types
        types = [MemoryType.CONVERSATION, MemoryType.FACT, MemoryType.CONVERSATION]
        for i, mtype in enumerate(types):
            entry = MemoryEntry(
                memory_id=f"mem_{i}",
                user_id="user_001",
                memory_type=mtype,
                content={"index": i},
                created_at=now,
                updated_at=now
            )
            store.store_memory(entry)

        conversations = store.get_user_memories(
            "user_001",
            memory_type=MemoryType.CONVERSATION
        )

        assert len(conversations) == 2

    def test_update_memory(self):
        """Test updating memory content"""
        store = InMemoryStore()
        now = datetime.now()

        entry = MemoryEntry(
            memory_id="mem_001",
            user_id="user_001",
            memory_type=MemoryType.PREFERENCE,
            content={"theme": "light"},
            created_at=now,
            updated_at=now
        )
        store.store_memory(entry)

        # Update
        assert store.update_memory("mem_001", {"theme": "dark"}) is True

        # Verify
        updated = store.get_memory("mem_001")
        assert updated.content == {"theme": "dark"}

    def test_delete_memory(self):
        """Test deleting a memory"""
        store = InMemoryStore()
        now = datetime.now()

        entry = MemoryEntry(
            memory_id="mem_001",
            user_id="user_001",
            memory_type=MemoryType.FACT,
            content={"fact": "test"},
            created_at=now,
            updated_at=now
        )
        store.store_memory(entry)

        # Delete
        assert store.delete_memory("mem_001") is True

        # Verify gone
        assert store.get_memory("mem_001") is None

    def test_delete_user_memories(self):
        """Test deleting all user memories"""
        store = InMemoryStore()
        now = datetime.now()

        # Store 3 memories
        for i in range(3):
            entry = MemoryEntry(
                memory_id=f"mem_{i}",
                user_id="user_001",
                memory_type=MemoryType.CONVERSATION,
                content={"index": i},
                created_at=now,
                updated_at=now
            )
            store.store_memory(entry)

        # Delete all
        count = store.delete_user_memories("user_001")

        assert count == 3
        assert len(store.get_user_memories("user_001")) == 0

    def test_search_memories(self):
        """Test searching memories"""
        store = InMemoryStore()
        now = datetime.now()

        entries = [
            {"message": "I love Python programming"},
            {"message": "JavaScript is also good"},
            {"message": "Python is my favorite"}
        ]

        for i, content in enumerate(entries):
            entry = MemoryEntry(
                memory_id=f"mem_{i}",
                user_id="user_001",
                memory_type=MemoryType.CONVERSATION,
                content=content,
                created_at=now,
                updated_at=now
            )
            store.store_memory(entry)

        results = store.search_memories("user_001", "Python")

        assert len(results) == 2

    def test_user_profile_operations(self):
        """Test profile save and retrieve"""
        store = InMemoryStore()

        profile = UserProfile(
            user_id="user_001",
            display_name="Test User",
            preferences={"language": "en"}
        )

        # Save
        assert store.save_user_profile(profile) is True

        # Retrieve
        retrieved = store.get_user_profile("user_001")
        assert retrieved is not None
        assert retrieved.display_name == "Test User"
        assert retrieved.preferences == {"language": "en"}

    def test_expired_memories_not_returned(self):
        """Test that expired memories are filtered out"""
        store = InMemoryStore()
        now = datetime.now()

        # Create expired memory
        entry = MemoryEntry(
            memory_id="mem_001",
            user_id="user_001",
            memory_type=MemoryType.CONTEXT,
            content={"session": "old"},
            created_at=now - timedelta(hours=2),
            updated_at=now - timedelta(hours=2),
            expires_at=now - timedelta(hours=1)  # Expired
        )
        store.store_memory(entry)

        memories = store.get_user_memories("user_001")

        assert len(memories) == 0


class TestUserMemory:
    """Tests for high-level UserMemory class"""

    def test_remember_and_recall(self):
        """Test basic remember and recall"""
        memory = UserMemory()

        memory_id = memory.remember(
            user_id="user_001",
            content={"data": "test"},
            memory_type=MemoryType.CONTEXT
        )

        assert memory_id.startswith("mem_")

        memories = memory.recall("user_001")
        assert len(memories) >= 1

    def test_remember_conversation(self):
        """Test storing conversation"""
        memory = UserMemory()

        memory.remember_conversation(
            user_id="user_001",
            role="user",
            message="Hello!"
        )
        memory.remember_conversation(
            user_id="user_001",
            role="assistant",
            message="Hi there!"
        )

        conversations = memory.recall_conversations("user_001")

        assert len(conversations) == 2
        assert conversations[0]["role"] in ["user", "assistant"]

    def test_remember_preference(self):
        """Test storing preferences"""
        memory = UserMemory()

        memory.remember_preference(
            user_id="user_001",
            key="theme",
            value="dark"
        )
        memory.remember_preference(
            user_id="user_001",
            key="language",
            value="python"
        )

        preferences = memory.recall_preferences("user_001")

        assert preferences.get("theme") == "dark"
        assert preferences.get("language") == "python"

    def test_remember_fact(self):
        """Test storing facts"""
        memory = UserMemory()

        memory.remember_fact(
            user_id="user_001",
            fact="User is a software developer"
        )
        memory.remember_fact(
            user_id="user_001",
            fact="User prefers Python"
        )

        facts = memory.recall_facts("user_001")

        assert len(facts) == 2
        assert "User is a software developer" in facts

    def test_get_context_summary(self):
        """Test getting user context summary"""
        memory = UserMemory()

        # Build up user history
        memory.update_profile("user_001", display_name="Test User")
        memory.remember_preference("user_001", "theme", "dark")
        memory.remember_fact("user_001", "Likes coding")
        memory.remember_conversation("user_001", "user", "Hello!")

        context = memory.get_context_summary("user_001")

        assert context["known"] is True
        assert context["display_name"] == "Test User"
        assert "theme" in context["preferences"]
        assert "Likes coding" in context["facts"]
        assert len(context["recent_conversations"]) >= 1

    def test_unknown_user_context(self):
        """Test context for unknown user"""
        memory = UserMemory()

        context = memory.get_context_summary("unknown_user")

        assert context["known"] is False
        assert context["user_id"] == "unknown_user"

    def test_forget_memory(self):
        """Test forgetting specific memory"""
        memory = UserMemory()

        memory_id = memory.remember(
            user_id="user_001",
            content={"secret": "data"}
        )

        assert memory.forget(memory_id) is True
        assert memory.forget(memory_id) is False  # Already gone

    def test_forget_user_gdpr(self):
        """Test GDPR-style full user deletion"""
        memory = UserMemory()

        # Create user data
        memory.remember_conversation("user_001", "user", "Message 1")
        memory.remember_conversation("user_001", "user", "Message 2")
        memory.remember_preference("user_001", "key", "value")
        memory.remember_fact("user_001", "Some fact")

        # Delete all
        count = memory.forget_user("user_001")

        assert count >= 4

        # Verify gone
        context = memory.get_context_summary("user_001")
        assert context["known"] is False

    def test_search(self):
        """Test memory search"""
        memory = UserMemory()

        memory.remember_conversation("user_001", "user", "I love quantum physics")
        memory.remember_conversation("user_001", "user", "Classical mechanics is okay")
        memory.remember_conversation("user_001", "user", "Quantum computing is cool")

        results = memory.search("user_001", "quantum")

        assert len(results) == 2

    def test_ttl_expiration(self):
        """Test memory TTL"""
        memory = UserMemory()

        # Store with very short TTL
        memory_id = memory.remember(
            user_id="user_001",
            content={"temporary": True},
            ttl_seconds=1
        )

        # Should exist immediately
        memories = memory.recall("user_001")
        assert any(m.memory_id == memory_id for m in memories)

        # Note: In production, expired memories would be filtered
        # The InMemoryStore filters on retrieval


class TestUtilityFunctions:
    """Tests for utility functions"""

    def test_get_user_hash(self):
        """Test user hash generation"""
        hash1 = get_user_hash("user@example.com")
        hash2 = get_user_hash("user@example.com")
        hash3 = get_user_hash("other@example.com")

        assert hash1 == hash2  # Deterministic
        assert hash1 != hash3  # Different input
        assert len(hash1) == 16  # Fixed length

    def test_create_memory_system_inmemory(self):
        """Test factory creates in-memory store"""
        memory = create_memory_system("memory")

        assert isinstance(memory, UserMemory)
        assert isinstance(memory.store, InMemoryStore)


class TestMemoryEnabledMixin:
    """Tests for MemoryEnabledMixin"""

    def test_init_memory(self):
        """Test mixin initialization"""
        class MockAgent(MemoryEnabledMixin):
            pass

        agent = MockAgent()
        agent.init_memory()

        assert hasattr(agent, "memory")
        assert agent._memory_enabled is True


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
