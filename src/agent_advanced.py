from __future__ import annotations

from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model

QUESTION_HINTS = (
    "?",
    "tên gì",
    "tên mình",
    "nhắc lại",
    "là gì",
    "ở đâu",
    "nghề",
    "style",
    "yêu thích",
    "nuôi con",
    "biết",
    "tóm tắt",
    "mô tả",
)


class AdvancedAgent:
    """Agent B: session + User.md + compact memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent and not self.force_offline:
            pass
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        messages = ctx.get("messages", [])
        msg_text = "\n".join(m["content"] for m in messages)
        return estimate_tokens(profile) + estimate_tokens(summary) + estimate_tokens(msg_text)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)
        lower = message.lower()

        if not any(hint in lower for hint in QUESTION_HINTS):
            return "Đã ghi nhớ thông tin của bạn."

        parts: list[str] = []
        labels = [
            ("name", "Tên"),
            ("location", "Nơi ở"),
            ("profession", "Nghề"),
            ("favorite_drink", "Đồ uống yêu thích"),
            ("favorite_food", "Món ăn yêu thích"),
            ("pet", "Thú cưng"),
            ("response_style", "Style trả lời"),
            ("interests", "Quan tâm"),
        ]
        for key, label in labels:
            value = facts.get(key)
            if value:
                parts.append(f"{label}: {value}")

        if parts:
            return " | ".join(parts)
        return "Chưa có thông tin ổn định trong User.md."

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        updates = extract_profile_updates(message)
        if updates:
            self.profile_store.upsert_facts(user_id, updates)

        self.compact_memory.append(thread_id, "user", message)

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        answer = self._offline_response(user_id, thread_id, message)
        agent_tokens = estimate_tokens(answer)

        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens

        return {
            "answer": answer,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver

            model = build_chat_model(self.config.model)
            self.langchain_agent = create_agent(
                model=model,
                tools=[],
                checkpointer=InMemorySaver(),
            )
        except Exception:
            self.langchain_agent = None
