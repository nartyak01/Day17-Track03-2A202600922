from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    hit = sum(1 for item in expected if item.lower() in answer.lower())
    if hit == 0:
        return 0.0
    if hit == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    if not answer.strip():
        return 0.0
    base = recall_points(answer, expected)
    bonus = 0.1 if len(answer) < 300 else 0.0
    return min(1.0, base + bonus)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
) -> BenchmarkRow:
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    memory_growth = 0
    compactions = 0

    for conv in conversations:
        user_id = conv["user_id"]
        thread_id = conv["id"]

        for turn in conv["turns"]:
            agent.reply(user_id, thread_id, turn)

        total_agent_tokens += agent.token_usage(thread_id)
        total_prompt_tokens += agent.prompt_token_usage(thread_id)

        if hasattr(agent, "memory_file_size"):
            memory_growth = max(memory_growth, agent.memory_file_size(user_id))
        compactions = max(compactions, agent.compaction_count(thread_id))

        for q in conv.get("recall_questions", []):
            recall_thread = f"{thread_id}-recall"
            resp = agent.reply(user_id, recall_thread, q["question"])
            recall_scores.append(recall_points(resp["answer"], q["expected_contains"]))
            quality_scores.append(heuristic_quality(resp["answer"], q["expected_contains"]))

    n = max(len(recall_scores), 1)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=sum(recall_scores) / n,
        response_quality=sum(quality_scores) / n,
        memory_growth_bytes=memory_growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    header = (
        "| Agent | Agent tokens | Prompt tokens | Recall | Quality "
        "| Memory (bytes) | Compactions |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r.agent_name} | {r.agent_tokens_only} | {r.prompt_tokens_processed} "
            f"| {r.recall_score:.2f} | {r.response_quality:.2f} "
            f"| {r.memory_growth_bytes} | {r.compactions} |"
        )
    return "\n".join(lines)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    standard = load_conversations(config.data_dir / "conversations.json")
    stress = load_conversations(config.data_dir / "advanced_long_context.json")

    print("## Standard benchmark")
    rows = [
        run_agent_benchmark(
            "Baseline",
            BaselineAgent(config, force_offline=True),
            standard,
            config,
        ),
        run_agent_benchmark(
            "Advanced",
            AdvancedAgent(config, force_offline=True),
            standard,
            config,
        ),
    ]
    print(format_rows(rows))

    print("\n## Long-context stress benchmark")
    rows2 = [
        run_agent_benchmark(
            "Baseline",
            BaselineAgent(config, force_offline=True),
            stress,
            config,
        ),
        run_agent_benchmark(
            "Advanced",
            AdvancedAgent(config, force_offline=True),
            stress,
            config,
        ),
    ]
    print(format_rows(rows2))


if __name__ == "__main__":
    main()
