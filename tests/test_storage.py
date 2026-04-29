"""Tests for SQLite-backed conversation storage."""

from __future__ import annotations

import os
import tempfile

import pytest

from ragpipe.server.storage import ConversationStore


@pytest.fixture
def store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    s = ConversationStore(db_path=path)
    yield s
    if os.path.exists(path):
        os.unlink(path)


def test_create_and_list_conversation(store):
    conv = store.create_conversation(title="Hello", model="gpt-5", provider="openai")
    assert conv["id"]
    assert conv["title"] == "Hello"
    assert conv["model"] == "gpt-5"

    convs = store.list_conversations()
    assert len(convs) == 1
    assert convs[0]["id"] == conv["id"]
    assert convs[0]["message_count"] == 0


def test_get_conversation_with_messages(store):
    conv = store.create_conversation(title="Chat 1")
    store.add_message(conv["id"], role="user", content="Hi")
    store.add_message(
        conv["id"], role="assistant", content="Hello there!",
        sources=[{"text": "src", "doc_id": "d", "score": 0.9, "rank": 1}],
        model="gpt-5", tokens_used=42, latency_ms=123.4,
    )

    full = store.get_conversation(conv["id"])
    assert full is not None
    assert len(full["messages"]) == 2
    assert full["messages"][0]["role"] == "user"
    assert full["messages"][0]["content"] == "Hi"
    assert full["messages"][1]["role"] == "assistant"
    assert full["messages"][1]["sources"][0]["doc_id"] == "d"
    assert full["messages"][1]["tokens_used"] == 42


def test_rename_conversation(store):
    conv = store.create_conversation(title="Old Name")
    ok = store.update_conversation_title(conv["id"], "New Name")
    assert ok
    fetched = store.get_conversation(conv["id"])
    assert fetched["title"] == "New Name"


def test_rename_missing_returns_false(store):
    assert store.update_conversation_title("does-not-exist", "x") is False


def test_delete_conversation_cascades_messages(store):
    conv = store.create_conversation(title="Chat")
    store.add_message(conv["id"], role="user", content="hi")
    store.add_message(conv["id"], role="assistant", content="hello")

    assert store.delete_conversation(conv["id"]) is True
    assert store.get_conversation(conv["id"]) is None


def test_delete_missing_conversation(store):
    assert store.delete_conversation("nonexistent") is False


def test_message_count_in_list(store):
    conv = store.create_conversation(title="Chat")
    store.add_message(conv["id"], role="user", content="1")
    store.add_message(conv["id"], role="assistant", content="2")
    store.add_message(conv["id"], role="user", content="3")

    convs = store.list_conversations()
    assert convs[0]["message_count"] == 3


def test_updated_at_changes_on_message_add(store):
    conv = store.create_conversation(title="Chat")
    initial_updated = conv["updated_at"]

    import time
    time.sleep(0.01)
    store.add_message(conv["id"], role="user", content="msg")

    refetched = store.list_conversations()[0]
    assert refetched["updated_at"] > initial_updated


def test_conversations_ordered_by_updated_desc(store):
    c1 = store.create_conversation(title="First")
    import time
    time.sleep(0.01)
    c2 = store.create_conversation(title="Second")

    convs = store.list_conversations()
    assert convs[0]["id"] == c2["id"]
    assert convs[1]["id"] == c1["id"]


def test_empty_sources_serialized_correctly(store):
    conv = store.create_conversation(title="Chat")
    store.add_message(conv["id"], role="assistant", content="Answer", sources=None)
    full = store.get_conversation(conv["id"])
    assert full["messages"][0]["sources"] == []
