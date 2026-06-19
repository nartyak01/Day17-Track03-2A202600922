from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session memory only, no persistent User.md."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    def _session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent and not self.force_offline:
            pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})

        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        answer = f"Đã ghi nhận trong phiên: {message[:120]}"
        agent_tokens = estimate_tokens(answer)

        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += agent_tokens

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
