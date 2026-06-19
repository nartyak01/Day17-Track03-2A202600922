from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    root = Path(__file__).resolve().parent.parent
    provider = ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)
    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=60,
        compact_keep_messages=2,
        model=provider,
        judge_model=provider,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    store.write_text("alice", "# User Profile\n\n- name: Alice\n")
    assert "Alice" in store.read_text("alice")
    assert store.edit_text("alice", "Alice", "Alicia")
    assert "Alicia" in store.read_text("alice")
    assert store.file_size("alice") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    mgr = CompactMemoryManager(threshold_tokens=40, keep_messages=2)
    for i in range(20):
        mgr.append("t1", "user", f"turn {i} " + "x" * 40)
    assert mgr.compaction_count("t1") >= 1


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    adv = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)

    adv.reply("u1", "thread-1", "Chào bạn, mình tên là DũngCT.")
    baseline.reply("u1", "thread-1", "Chào bạn, mình tên là DũngCT.")

    adv_answer = adv.reply("u1", "thread-2", "Mình tên gì?")["answer"]
    base_answer = baseline.reply("u1", "thread-2", "Mình tên gì?")["answer"]

    assert "DũngCT" in adv_answer
    assert "DũngCT" not in base_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    for i in range(30):
        msg = f"turn {i} " + "benchmark memory compaction " * 5
        baseline.reply("u1", "long", msg)
        advanced.reply("u1", "long", msg)

    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
