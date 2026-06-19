from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PROFILE = "# User Profile\n\n"


def estimate_tokens(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


def _slug(user_id: str) -> str:
    return re.sub(r"[^\w\-]+", "_", user_id.strip().lower()) or "default"


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        return self.root_dir / _slug(user_id) / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return DEFAULT_PROFILE
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def upsert_facts(self, user_id: str, facts: dict[str, str]) -> None:
        content = self.read_text(user_id)
        for key, value in facts.items():
            line = f"- {key}: {value}"
            pattern = rf"^- {re.escape(key)}:.*$"
            if re.search(pattern, content, flags=re.MULTILINE):
                content = re.sub(pattern, line, content, count=1, flags=re.MULTILINE)
            else:
                content = content.rstrip() + f"\n{line}\n"
        self.write_text(user_id, content)

    def facts(self, user_id: str) -> dict[str, str]:
        content = self.read_text(user_id)
        facts: dict[str, str] = {}
        for match in re.finditer(r"^- (\w+): (.+)$", content, flags=re.MULTILINE):
            facts[match.group(1)] = match.group(2).strip()
        return facts


def extract_profile_updates(message: str) -> dict[str, str]:
    text = message.strip()
    if not text:
        return {}

    lower = text.lower()
    if "câu đùa" in lower or "chỉ là nơi mình vừa bay" in lower:
        return {}

    facts: dict[str, str] = {}
    patterns: dict[str, list[str]] = {
        "name": [
            r"mình tên là\s+([^.,\n]+)",
            r"tên mình là\s+([^.,\n]+)",
            r"tên\s+(DũngCT(?:\s+Stress)?)",
            r"nhắc lại: tên\s+([^.,\n]+)",
        ],
        "location": [
            r"nơi ở hiện tại là\s+([^.,\n]+)",
            r"đang làm việc ở\s+([^.,\n]+)",
            r"hiện tại mình đang ở\s+([^.,\n]+)",
            r"mình vẫn ở\s+([^.,\n]+)",
            r"cập nhật từ [^ ]+ sang\s+([^.,\n]+)",
            r"chuyển.*?sang\s+([^.,\n]+)",
            r"mình (?:ở|đang ở|hiện ở)\s+([^.,\n]+)",
        ],
        "profession": [
            r"nghề nghiệp vẫn là\s+([^.,\n]+)",
            r"nghề nghiệp hiện tại vẫn là\s+([^.,\n]+)",
            r"giờ chuyển sang\s+([^.,\n]+)",
            r"đang làm\s+(backend engineer|MLOps engineer)",
            r"làm\s+(backend engineer|MLOps engineer)",
            r"nghề\s+(MLOps engineer)",
        ],
        "favorite_drink": [
            r"đồ uống yêu thích là\s+([^.,\n]+)",
            r"uống\s+(cà phê sữa đá)",
        ],
        "favorite_food": [
            r"món ăn yêu thích là\s+([^.,\n]+)",
        ],
        "pet": [
            r"nuôi (?:một )?(?:bé )?(corgi)(?: tên\s+([^.,\n]+))?",
            r"con (corgi)(?: tên\s+([^.,\n]+))?",
        ],
        "response_style": [
            r"trả lời ((?:ngắn gọn|thành 3 bullet)[^.]*)",
            r"muốn bạn trả lời ((?:ngắn gọn|thành 3 bullet)[^.]*)",
            r"style trả lời[^:]*:\s*([^.\n]+)",
            r"(\d+ bullet[^.]*)",
        ],
        "interests": [
            r"quan tâm nhiều đến\s+([^.\n]+)",
            r"thích\s+(Python[^.\n]*)",
        ],
    }

    for key, regex_list in patterns.items():
        for pattern in regex_list:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                if key == "pet" and m.lastindex and m.lastindex >= 1:
                    breed = m.group(1).strip()
                    name = m.group(2).strip() if m.lastindex >= 2 and m.group(2) else ""
                    facts[key] = f"{breed} tên {name}".strip() if name else breed
                else:
                    value = m.group(1).strip()
                    value = re.sub(r"\s{2,}", " ", value)
                    facts[key] = value
                break

    if "profession" in facts:
        prof = facts["profession"].lower()
        if "mlops" in prof:
            facts["profession"] = "MLOps engineer"
        elif "backend" in prof:
            facts["profession"] = "backend engineer"

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    picked = messages[:max_items]
    parts = [f"{m['role']}: {m['content']}" for m in picked]
    return " | ".join(parts)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def _total_tokens(self, thread: dict[str, object]) -> int:
        messages = thread["messages"]
        summary = thread["summary"]
        msg_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        summary_tokens = estimate_tokens(summary) if summary else 0
        return msg_tokens + summary_tokens

    def _compact(self, thread: dict[str, object]) -> None:
        messages = thread["messages"]
        if len(messages) <= self.keep_messages:
            return
        old = messages[:-self.keep_messages]
        thread["summary"] = (
            (thread["summary"] + " | " if thread["summary"] else "")
            + summarize_messages(old)
        )
        thread["messages"] = messages[-self.keep_messages:]
        thread["compactions"] = int(thread["compactions"]) + 1

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._ensure(thread_id)
        thread["messages"].append({"role": role, "content": content})
        while (
            self._total_tokens(thread) > self.threshold_tokens
            and len(thread["messages"]) > self.keep_messages
        ):
            self._compact(thread)

    def context(self, thread_id: str) -> dict[str, object]:
        return self._ensure(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self._ensure(thread_id)["compactions"])
