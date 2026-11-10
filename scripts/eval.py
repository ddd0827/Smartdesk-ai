from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings  # noqa: E402
from app.rag.index import RetrievalIndex, RetrievedChunk  # noqa: E402
from app.rag.ingest import DocumentService  # noqa: E402
from app.rag.vectors import EmbeddingError, build_embedder  # noqa: E402
from app.store.db import Database  # noqa: E402
from app.store.repo import Repository  # noqa: E402

SearchFn = Callable[[str, int], Awaitable[list[RetrievedChunk]]]


def _rank_of(results: list[RetrievedChunk], expected: set[str]) -> int | None:
    for index, item in enumerate(results, start=1):
        if item.filename in expected:
            return index
    return None


def _answer_rank(
    results: list[RetrievedChunk], keywords: set[str]
) -> int | None:
    if not keywords:
        return None
    for index, item in enumerate(results, start=1):
        if all(keyword in item.content for keyword in keywords):
            return index
    return None


def _fmt_rank(rank: int | None) -> str:
    return "—" if rank is None else str(rank)


def _escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


async def evaluate(questions_path: Path) -> tuple[dict[str, Any], str]:
    data = json.loads(questions_path.read_text(encoding="utf-8"))
    questions: list[dict[str, Any]] = data["questions"]
    k = int(data.get("k", 3))

    settings = Settings(
        data_dir=Path(tempfile.mkdtemp(prefix="smartdesk-eval-")),
        llm_provider="mock",
        auto_seed=False,
    )
    db = Database(settings.db_path)
    await db.init()
    repo = Repository(db)
    embedder = build_embedder(settings)
    index = RetrievalIndex(repo, embedder, settings)
    service = DocumentService(repo, embedder, index, settings)
    seed_result = await service.seed(force=False)
    await index.ensure_ready()
    if index.size == 0:
        raise RuntimeError(
            "索引为空。如果切换了 embedding 配置，请先重建索引后再评估。"
        )

    methods: dict[str, SearchFn] = {
        "BM25": index.search_bm25,
        "Hash 向量": index.search_vector,
        "混合检索": index.search_hybrid,
    }
    for search in methods.values():
        await search("退换货 物流 保修", 5)

    per_method: dict[str, list[dict[str, Any]]] = {name: [] for name in methods}
    for question in questions:
        expected = set(question.get("expected_docs") or [question["expected_doc"]])
        keywords = set(question.get("keywords") or [])
        for name, search in methods.items():
            started = time.perf_counter()
            results = await search(question["question"], 10)
            latency_ms = (time.perf_counter() - started) * 1000
            per_method[name].append(
                {
                    "id": question["id"],
                    "question": question["question"],
                    "expected": sorted(expected),
                    "rank": _answer_rank(results, keywords),
                    "doc_rank": _rank_of(results, expected),
                    "top": [item.filename for item in results[:k]],
                    "latency_ms": latency_ms,
                }
            )

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [row for row in rows if row["rank"] is not None]
        return {
            "recall": sum(1 for row in valid if row["rank"] <= k) / len(rows),
            "top1": sum(1 for row in valid if row["rank"] == 1) / len(rows),
            "mrr": sum(1.0 / row["rank"] for row in valid) / len(rows),
            "doc_recall": sum(
                1
                for row in rows
                if row["doc_rank"] is not None and row["doc_rank"] <= k
            )
            / len(rows),
            "median_latency_ms": statistics.median(
                row["latency_ms"] for row in rows
            ),
            "hits": len(valid),
        }

    summary = {name: summarize(rows) for name, rows in per_method.items()}

    wins = ties = losses = 0
    for bm25_row, hybrid_row in zip(
        per_method["BM25"], per_method["混合检索"], strict=True
    ):
        a, b = bm25_row["rank"], hybrid_row["rank"]
        if a is None and b is None:
            ties += 1
        elif b is None:
            losses += 1
        elif a is None or b < a:
            wins += 1
        elif b == a:
            ties += 1
        else:
            losses += 1

    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    method_labels = {
        "BM25": "BM25-only",
        "Hash 向量": "HashNgramVectorizer-only",
        "混合检索": "Hybrid (RRF)",
    }
    lines = [
        "# 检索评估报告",
        "",
        f"生成时间：{generated}",
        "",
        "## 设置",
        "",
        f"- 语料：{seed_result.documents_added} 篇演示文档、{index.size} 个分块",
        f"- 评估集：{len(questions)} 题，运行前冻结（`eval/questions.json`）",
        f"- 主指标：**答案片段命中**——只有检索到同时包含该题全部关键词的单个分块才算命中；"
        f"统计 Recall@{k}、Top-1、MRR@10",
        f"- 辅助指标：文档级 Recall@{k}（命中正确文档即算，即使分块未包含完整答案）",
        f"- 融合：RRF，k={settings.rrf_k}；检索候选数={settings.retrieval_candidates}",
        f"- 向量方式：`{embedder.signature}`",
        "",
        "## 结果",
        "",
        f"| 方法 | 答案 Recall@{k} | 答案 Top-1 | MRR@10 | 文档 Recall@{k} | 中位耗时 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in summary.items():
        lines.append(
            f"| {method_labels[name]} | {metrics['recall']:.3f} | "
            f"{metrics['top1']:.3f} | {metrics['mrr']:.3f} | "
            f"{metrics['doc_recall']:.3f} | "
            f"{metrics['median_latency_ms']:.2f} ms |"
        )

    lines += [
        "",
        "## 混合检索相对 BM25 的逐题对比",
        "",
        f"- 胜：{wins}，平：{ties}，负：{losses}",
        "",
        "| 题目 | 期望文档 | BM25 答案排名 | Hash 向量答案排名 | 混合答案排名 |",
        "|---|---|---:|---:|---:|",
    ]
    for bm25_row, vector_row, hybrid_row in zip(
        per_method["BM25"],
        per_method["Hash 向量"],
        per_method["混合检索"],
        strict=True,
    ):
        lines.append(
            f"| {bm25_row['id']} {_escape(bm25_row['question'])} | "
            f"{_escape('、'.join(bm25_row['expected']))} | "
            f"{_fmt_rank(bm25_row['rank'])} | {_fmt_rank(vector_row['rank'])} | "
            f"{_fmt_rank(hybrid_row['rank'])} |"
        )

    lines += [
        "",
        "## 口径与局限",
        "",
        (
            "- `HashNgramVectorizer` 是固定种子的字符 n-gram 哈希词法向量，"
            "**不是语义 embedding 模型**。"
        ),
        (
            "  没有配置 BGE-M3 等远程 embedding 时，本报告只能说明词法级混合检索管线可用，"
            "  不能据此宣称语义检索效果。"
        ),
        "- 答案片段要求单个分块同时包含全部关键词；如果切分把答案拆到两个分块，会判为未命中。",
        "  这比文档级命中更严格，也同时暴露切分与检索的问题。",
        "- 语料只有 8 篇演示文档，题目由项目作者自标注，属于小样本方向性评估，不做显著性结论。",
        (
            "- 题目在运行前冻结；若后续调整切分、RRF 参数或语料，应重新记录指标，"
            "不能只保留最好看的一次。"
        ),
        (
            "- 真实回答的引用正确率、拒答行为与用户满意度需要配置真实 LLM 后"
            "按 `docs/demo-script.md` 人工检查。"
        ),
        "",
        "## 复现",
        "",
        "```bash",
        "python scripts/eval.py",
        "```",
        "",
    ]

    metrics = {
        "generated_at": generated,
        "corpus": {
            "documents": await repo.count_documents(),
            "chunks": index.size,
            "embedding": embedder.signature,
        },
        "questions": len(questions),
        "summary": summary,
        "hybrid_vs_bm25": {"wins": wins, "ties": ties, "losses": losses},
        "per_method": per_method,
    }
    return metrics, "\n".join(lines)


async def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="SmartDesk 检索评估")
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "eval" / "questions.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "docs" / "eval-report.md",
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    try:
        metrics, report = await evaluate(args.questions)
    except EmbeddingError as exc:
        print(f"embedding 配置失败：{exc}")
        return 2
    except RuntimeError as exc:
        print(f"评估失败：{exc}")
        return 2

    print("=== 检索评估结果 ===")
    for name, values in metrics["summary"].items():
        print(
            f"{name}: 答案 Recall@3={values['recall']:.3f}, "
            f"答案 Top-1={values['top1']:.3f}, MRR@10={values['mrr']:.3f}, "
            f"文档 Recall@3={values['doc_recall']:.3f}, "
            f"中位耗时={values['median_latency_ms']:.2f} ms"
        )
    comparison = metrics["hybrid_vs_bm25"]
    print(
        "混合 vs BM25："
        f"胜 {comparison['wins']} / 平 {comparison['ties']} / 负 {comparison['losses']}"
    )
    if not args.no_write:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"报告已写入 {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
