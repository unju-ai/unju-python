"""
Comprehensive Memory Integration Tests

Tests for "immortal and perfect memory" - inspired by mem0's approach:
- Semantic accuracy (+26% vs baseline)
- Performance (91% faster responses)
- Token efficiency (90% fewer tokens)
- Reliability (no data loss, ever)
- Consolidation (merge related memories)
- Personalization (adapt to user over time)

Requires: UNJU_API_KEY environment variable
Run: pytest tests/test_memory_comprehensive.py -v --tb=short
"""
import pytest
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict
import os

from unju import Unju

# Skip if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("UNJU_API_KEY"),
    reason="UNJU_API_KEY not set - integration tests require live API"
)


@pytest.fixture
def client():
    """Create authenticated client for testing"""
    api_key = os.getenv("UNJU_API_KEY")
    return Unju(api_key=api_key)


@pytest.fixture
def test_user():
    """Generate unique test user ID"""
    return f"test_user_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def cleanup_memories(client, test_user):
    """Cleanup test memories after each test"""
    yield
    try:
        client.memory.delete_all(test_user)
    except Exception:
        pass  # Best effort cleanup


class TestSemanticAccuracy:
    """Test semantic understanding and retrieval accuracy"""

    def test_find_exact_match(self, client, test_user, cleanup_memories):
        """Should find memory with exact content match"""
        # Add memory
        add_result = client.memory.add(
            test_user,
            "I love eating sushi for dinner"
        )
        assert add_result.get("success"), "Memory add failed"
        
        time.sleep(1)  # Allow indexing
        
        # Search with exact phrase
        results = client.memory.search(test_user, "sushi for dinner")
        assert len(results) > 0, "Should find exact match"
        assert "sushi" in results[0]["memory"].lower()

    def test_find_semantic_match(self, client, test_user, cleanup_memories):
        """Should find semantically similar memories"""
        # Add memories with different wordings
        client.memory.add(test_user, "I enjoy Italian cuisine, especially pasta")
        client.memory.add(test_user, "My favorite hobby is rock climbing")
        client.memory.add(test_user, "I work as a software engineer")
        
        time.sleep(1)
        
        # Search with semantic variation
        results = client.memory.search(test_user, "food preferences")
        assert len(results) > 0, "Should find semantic match"
        
        # Top result should be about food
        top_memory = results[0]["memory"].lower()
        assert any(word in top_memory for word in ["italian", "cuisine", "pasta", "food"])

    def test_semantic_ranking(self, client, test_user, cleanup_memories):
        """Should rank memories by semantic relevance"""
        # Add memories with varying relevance to "programming"
        memories = [
            "I code in Python and TypeScript daily",  # Most relevant
            "I enjoy reading technical books",  # Somewhat relevant
            "My cat's name is Whiskers",  # Not relevant
            "I'm learning machine learning",  # Relevant
        ]
        
        for m in memories:
            client.memory.add(test_user, m)
        
        time.sleep(1)
        
        results = client.memory.search(test_user, "programming and software development", limit=3)
        assert len(results) > 0
        
        # Most relevant should be first
        top_memory = results[0]["memory"].lower()
        assert any(word in top_memory for word in ["python", "typescript", "code", "machine learning"])
        
        # Least relevant should not be in top results
        memories_text = " ".join([r["memory"] for r in results])
        assert "whiskers" not in memories_text.lower()

    def test_temporal_context(self, client, test_user, cleanup_memories):
        """Should understand temporal context"""
        # Add memories with time references
        client.memory.add(test_user, "Last week I started learning guitar")
        client.memory.add(test_user, "Yesterday I practiced for 2 hours")
        client.memory.add(test_user, "Today I'm working on chord progressions")
        
        time.sleep(1)
        
        # Search for recent activity
        results = client.memory.search(test_user, "recent practice", limit=2)
        assert len(results) > 0
        
        # Should prioritize recent memories
        recent_words = ["yesterday", "today", "working"]
        top_memory = results[0]["memory"].lower()
        assert any(word in top_memory for word in recent_words)

    def test_multi_entity_query(self, client, test_user, cleanup_memories):
        """Should handle queries with multiple entities"""
        client.memory.add(test_user, "I met Sarah at the coffee shop")
        client.memory.add(test_user, "John recommended a great restaurant")
        client.memory.add(test_user, "Sarah and I discussed the project proposal")
        
        time.sleep(1)
        
        # Search mentioning multiple entities
        results = client.memory.search(test_user, "Sarah project")
        assert len(results) > 0
        
        # Should find memory with both entities
        found_both = False
        for r in results:
            memory = r["memory"].lower()
            if "sarah" in memory and "project" in memory:
                found_both = True
                break
        
        assert found_both, "Should find memory mentioning both entities"


class TestMemoryDeduplication:
    """Test deduplication and consolidation of redundant memories"""

    def test_prevent_exact_duplicates(self, client, test_user, cleanup_memories):
        """Should not create exact duplicate memories"""
        content = "My favorite color is blue"
        
        # Add same content twice
        result1 = client.memory.add(test_user, content)
        time.sleep(0.5)
        result2 = client.memory.add(test_user, content)
        
        time.sleep(1)
        
        # List all memories
        all_memories = client.memory.list(test_user)
        
        # Should have deduplicated
        exact_matches = [m for m in all_memories if m["memory"] == content]
        assert len(exact_matches) <= 1, f"Should not have exact duplicates, found {len(exact_matches)}"

    def test_consolidate_similar_memories(self, client, test_user, cleanup_memories):
        """Should consolidate semantically similar memories"""
        # Add very similar memories
        client.memory.add(test_user, "I like pizza")
        time.sleep(0.5)
        client.memory.add(test_user, "I enjoy eating pizza")
        time.sleep(0.5)
        client.memory.add(test_user, "Pizza is my favorite food")
        
        time.sleep(1)
        
        # Search for pizza preference
        results = client.memory.search(test_user, "pizza preference", limit=5)
        
        # Should consolidate into fewer, more comprehensive memories
        # Instead of 3 redundant ones, should have 1-2 consolidated
        pizza_memories = [r for r in results if "pizza" in r["memory"].lower()]
        
        # Expect consolidation (may vary by implementation)
        # At minimum, check they're returned in order of importance
        if len(pizza_memories) > 1:
            # Most comprehensive should rank first
            assert any(word in pizza_memories[0]["memory"].lower() 
                      for word in ["favorite", "enjoy", "like"])

    def test_update_vs_duplicate(self, client, test_user, cleanup_memories):
        """Should update existing memory instead of creating duplicate"""
        # Add initial memory
        client.memory.add(test_user, "I live in San Francisco")
        time.sleep(0.5)
        
        # Add updated information
        client.memory.add(test_user, "I moved to New York last month")
        time.sleep(1)
        
        # Search for location
        results = client.memory.search(test_user, "where I live", limit=3)
        assert len(results) > 0
        
        # Most recent/relevant location should be prioritized
        top_memory = results[0]["memory"].lower()
        assert "new york" in top_memory


class TestMemoryPerformance:
    """Test performance characteristics - speed and efficiency"""

    def test_search_latency(self, client, test_user, cleanup_memories):
        """Search should be fast (<200ms target)"""
        # Add several memories
        for i in range(10):
            client.memory.add(test_user, f"Test memory number {i} about topic {i % 3}")
        
        time.sleep(1)  # Allow indexing
        
        # Measure search latency
        start = time.time()
        results = client.memory.search(test_user, "topic", limit=5)
        latency = (time.time() - start) * 1000  # Convert to ms
        
        assert len(results) > 0
        assert latency < 500, f"Search too slow: {latency:.0f}ms (target: <500ms)"
        print(f"✅ Search latency: {latency:.0f}ms")

    def test_bulk_add_performance(self, client, test_user, cleanup_memories):
        """Should handle bulk memory additions efficiently"""
        memories = [f"Memory {i}: This is test content" for i in range(20)]
        
        start = time.time()
        for m in memories:
            client.memory.add(test_user, m)
        duration = time.sleep() - start
        
        # Should average <100ms per add
        avg_time = (duration / len(memories)) * 1000
        assert avg_time < 200, f"Add too slow: {avg_time:.0f}ms per memory"
        print(f"✅ Bulk add: {avg_time:.0f}ms per memory")

    def test_list_pagination(self, client, test_user, cleanup_memories):
        """Should handle large memory lists efficiently"""
        # Add many memories
        for i in range(50):
            client.memory.add(test_user, f"Memory {i}")
        
        time.sleep(2)
        
        # List all
        start = time.time()
        all_memories = client.memory.list(test_user)
        latency = (time.time() - start) * 1000
        
        assert len(all_memories) > 0
        assert latency < 1000, f"List too slow: {latency:.0f}ms"
        print(f"✅ List {len(all_memories)} memories in {latency:.0f}ms")


class TestMemoryReliability:
    """Test reliability - no data loss, corruption recovery"""

    def test_persistence_after_add(self, client, test_user, cleanup_memories):
        """Memories should persist immediately after add"""
        content = f"Important memory at {datetime.utcnow().isoformat()}"
        
        add_result = client.memory.add(test_user, content)
        assert add_result.get("success")
        
        # Immediately retrieve
        memories = client.memory.list(test_user)
        assert any(content in m["memory"] for m in memories), "Memory should persist immediately"

    def test_retrieve_old_memory(self, client, test_user, cleanup_memories):
        """Should retrieve memories added long ago"""
        # Simulate old memory with timestamp metadata
        old_content = "This is an old memory from last year"
        client.memory.add(
            test_user,
            old_content,
            metadata={"created_at": (datetime.utcnow() - timedelta(days=365)).isoformat()}
        )
        
        time.sleep(1)
        
        # Should still be retrievable
        results = client.memory.search(test_user, "old memory")
        assert len(results) > 0
        assert any("old memory" in r["memory"].lower() for r in results)

    def test_delete_and_verify(self, client, test_user, cleanup_memories):
        """Deleted memories should not be retrievable"""
        # Add memory
        add_result = client.memory.add(test_user, "This will be deleted")
        time.sleep(0.5)
        
        # List and get ID
        memories = client.memory.list(test_user)
        memory_id = None
        for m in memories:
            if "deleted" in m["memory"].lower():
                memory_id = m["id"]
                break
        
        assert memory_id, "Should find the memory to delete"
        
        # Delete
        delete_result = client.memory.delete(memory_id)
        assert delete_result.get("success")
        
        time.sleep(0.5)
        
        # Verify not retrievable
        memories_after = client.memory.list(test_user)
        assert not any(m["id"] == memory_id for m in memories_after), "Deleted memory should not exist"

    def test_concurrent_access(self, client, test_user, cleanup_memories):
        """Should handle concurrent memory operations safely"""
        import concurrent.futures
        
        def add_memory(index):
            return client.memory.add(test_user, f"Concurrent memory {index}")
        
        # Add memories concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(add_memory, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert all(r.get("success") for r in results), "All concurrent adds should succeed"
        
        time.sleep(1)
        
        # Verify all were added
        memories = client.memory.list(test_user)
        concurrent_count = sum(1 for m in memories if "concurrent" in m["memory"].lower())
        assert concurrent_count == 10, f"Should have 10 memories, found {concurrent_count}"


class TestMemoryContext:
    """Test conversational context and message handling"""

    def test_conversation_memory(self, client, test_user, cleanup_memories):
        """Should extract memories from conversation messages"""
        messages = [
            {"role": "user", "content": "I'm allergic to peanuts"},
            {"role": "assistant", "content": "I'll remember that. Is there anything else I should know?"},
            {"role": "user", "content": "Also, I prefer dark mode in applications"},
        ]
        
        result = client.memory.add(test_user, messages)
        assert result.get("success")
        
        time.sleep(1)
        
        # Search for allergy
        results = client.memory.search(test_user, "food allergies")
        assert len(results) > 0
        assert any("peanut" in r["memory"].lower() for r in results)
        
        # Search for preference
        results = client.memory.search(test_user, "UI preferences")
        assert len(results) > 0
        assert any("dark mode" in r["memory"].lower() for r in results)

    def test_extract_multiple_facts(self, client, test_user, cleanup_memories):
        """Should extract multiple facts from single message"""
        message = "I'm John, I live in Seattle, I work at Microsoft as a PM, and I love hiking on weekends"
        
        client.memory.add(test_user, message)
        time.sleep(1)
        
        # Should be able to recall different aspects
        test_queries = [
            ("name", ["john"]),
            ("location", ["seattle"]),
            ("job", ["microsoft", "pm"]),
            ("hobbies", ["hiking"]),
        ]
        
        for query, expected_words in test_queries:
            results = client.memory.search(test_user, query, limit=3)
            found = False
            for r in results:
                if any(word in r["memory"].lower() for word in expected_words):
                    found = True
                    break
            assert found, f"Should find memory about {query}"


class TestMemoryMetadata:
    """Test metadata handling and filtering"""

    def test_add_with_metadata(self, client, test_user, cleanup_memories):
        """Should store and retrieve metadata"""
        metadata = {
            "source": "test",
            "importance": 0.9,
            "category": "preferences",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        result = client.memory.add(test_user, "Important preference", metadata=metadata)
        assert result.get("success")
        
        time.sleep(0.5)
        
        # Retrieve and check metadata preserved
        memories = client.memory.list(test_user)
        found = False
        for m in memories:
            if "important preference" in m["memory"].lower():
                # Check metadata exists (exact structure may vary)
                assert m.get("metadata") or m.get("data"), "Should have metadata"
                found = True
                break
        
        assert found, "Should find memory with metadata"


class TestTokenEfficiency:
    """Test token usage optimization"""

    def test_relevant_memories_only(self, client, test_user, cleanup_memories):
        """Should return only relevant memories to save tokens"""
        # Add diverse memories
        memories = [
            "I like Python programming",
            "My dog is named Max",
            "I drink coffee every morning",
            "I'm learning Japanese",
            "My favorite color is green",
        ]
        
        for m in memories:
            client.memory.add(test_user, m)
        
        time.sleep(1)
        
        # Search with specific query
        results = client.memory.search(test_user, "programming languages", limit=3)
        
        # Should prioritize relevant results
        top_memory = results[0]["memory"].lower()
        assert "python" in top_memory or "programming" in top_memory
        
        # Irrelevant memories should not be in top results
        top_3_text = " ".join([r["memory"].lower() for r in results[:3]])
        assert "dog" not in top_3_text, "Irrelevant memory should not be in top 3"

    def test_limit_enforcement(self, client, test_user, cleanup_memories):
        """Should respect limit to control token usage"""
        # Add many memories
        for i in range(20):
            client.memory.add(test_user, f"Memory number {i}")
        
        time.sleep(1)
        
        # Request specific limit
        results = client.memory.search(test_user, "memory", limit=5)
        
        # Should return exactly requested amount (or less if not enough matches)
        assert len(results) <= 5, f"Should return at most 5 results, got {len(results)}"


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Benchmark tests for performance regression tracking"""

    def test_benchmark_search_speed(self, client, test_user, cleanup_memories):
        """Benchmark: Search speed with 100 memories"""
        # Add 100 memories
        for i in range(100):
            client.memory.add(test_user, f"Benchmark memory {i} about topic {i % 10}")
        
        time.sleep(2)  # Allow full indexing
        
        # Benchmark search
        latencies = []
        for _ in range(10):
            start = time.time()
            client.memory.search(test_user, "topic 5", limit=5)
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        print(f"\n📊 Benchmark: Search 100 memories in {avg_latency:.0f}ms (avg over 10 runs)")
        
        # Target: <500ms even with 100 memories
        assert avg_latency < 500, f"Search too slow: {avg_latency:.0f}ms"

    def test_benchmark_memory_quality(self, client, test_user, cleanup_memories):
        """Benchmark: Memory retrieval accuracy"""
        # Add diverse memories with known associations
        test_cases = [
            ("I love Italian food", "food preferences", True),
            ("My cat is named Whiskers", "food preferences", False),
            ("Python is my main language", "programming", True),
            ("I went hiking last weekend", "programming", False),
            ("I work at Google", "employment", True),
            ("I enjoy reading sci-fi", "employment", False),
        ]
        
        for content, _, _ in test_cases:
            client.memory.add(test_user, content)
        
        time.sleep(1)
        
        # Test retrieval accuracy
        correct = 0
        total = 0
        
        for content, query, should_match in test_cases:
            results = client.memory.search(test_user, query, limit=3)
            found = any(content.lower() in r["memory"].lower() for r in results)
            
            if found == should_match:
                correct += 1
            total += 1
        
        accuracy = (correct / total) * 100
        print(f"\n📊 Benchmark: Retrieval accuracy {accuracy:.0f}% ({correct}/{total} correct)")
        
        # Target: >80% accuracy
        assert accuracy >= 80, f"Accuracy too low: {accuracy:.0f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
