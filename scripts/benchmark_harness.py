#!/usr/bin/env python3
"""
Research-style evaluation harness for Recursive Knowledge Engine.

Usage:
  python scripts/benchmark_harness.py \
    --api-base http://127.0.0.1:8000/api \
    --queries-file ./eval_queries.jsonl \
    --docs-dir ./eval_docs \
    --output ./benchmark_report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from uuid import uuid4

import requests


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".log",
    ".ini",
    ".cfg",
    ".conf",
    ".toml",
    ".env",
    ".rtf",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
}


@dataclass
class EvalItem:
    question: str
    expected_keywords: List[str]


def _read_eval_items(path: Path) -> List[EvalItem]:
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    items: List[EvalItem] = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            items.append(
                EvalItem(
                    question=str(payload.get("question", "")).strip(),
                    expected_keywords=[str(k).strip().lower() for k in payload.get("expected_keywords", []) if str(k).strip()],
                )
            )
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload:
            items.append(
                EvalItem(
                    question=str(row.get("question", "")).strip(),
                    expected_keywords=[str(k).strip().lower() for k in row.get("expected_keywords", []) if str(k).strip()],
                )
            )

    items = [item for item in items if len(item.question) >= 3]
    if not items:
        raise ValueError("No valid questions found in queries file.")
    return items


def _batched(items: List[Path], size: int) -> Iterable[List[Path]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _upload_docs(api_base: str, docs_dir: Path, batch_size: int = 8) -> Dict:
    docs = [
        path
        for path in sorted(docs_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not docs:
        return {"uploaded_files": 0, "failed_files": 0, "batch_errors": []}

    uploaded = 0
    failed = 0
    batch_errors: List[str] = []
    endpoint = f"{api_base.rstrip('/')}/upload/batch"

    for batch in _batched(docs, batch_size):
        files_payload = [("files", (doc.name, doc.read_bytes(), "application/octet-stream")) for doc in batch]
        response = requests.post(endpoint, files=files_payload, timeout=180)
        if not response.ok:
            batch_errors.append(f"HTTP {response.status_code}: {response.text[:200]}")
            failed += len(batch)
            continue
        data = response.json()
        for item in data.get("items", []):
            if item.get("error"):
                failed += 1
            else:
                uploaded += 1

    return {
        "uploaded_files": uploaded,
        "failed_files": failed,
        "batch_errors": batch_errors,
    }


def _keyword_recall(answer: str, expected_keywords: List[str]) -> Optional[float]:
    if not expected_keywords:
        return None
    normalized = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword in normalized)
    return round(hits / len(expected_keywords), 3)


def run_benchmark(
    api_base: str,
    items: List[EvalItem],
    session_id: str,
    top_k: int,
    max_iterations: int,
    fast_mode: bool,
    deterministic_mode: bool,
    answer_verbosity: str,
    challenge_mode: bool,
) -> Dict:
    endpoint = f"{api_base.rstrip('/')}/query"
    results: List[Dict] = []

    for idx, item in enumerate(items, start=1):
        started = time.perf_counter()
        response = requests.post(
            endpoint,
            json={
                "question": item.question,
                "session_id": session_id,
                "top_k": top_k,
                "max_iterations": max_iterations,
                "fast_mode": fast_mode,
                "deterministic_mode": deterministic_mode,
                "answer_verbosity": answer_verbosity,
                "challenge_mode": challenge_mode,
            },
            timeout=240,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if not response.ok:
            results.append(
                {
                    "index": idx,
                    "question": item.question,
                    "ok": False,
                    "http_status": response.status_code,
                    "error": response.text[:320],
                    "elapsed_ms_client": elapsed_ms,
                }
            )
            continue

        payload = response.json()
        answer = str(payload.get("final_answer", ""))
        loops = payload.get("loops", []) or []
        groundedness = float((loops[payload.get("best_iteration", 1) - 1].get("groundedness", 0.0)) if loops else 0.0)

        results.append(
            {
                "index": idx,
                "question": item.question,
                "ok": True,
                "response_id": payload.get("response_id"),
                "confidence": float(payload.get("confidence", 0.0)),
                "coverage": float(payload.get("answer_coverage", 0.0)),
                "groundedness": groundedness,
                "latency_ms_server": int(payload.get("total_duration_ms", 0)),
                "latency_ms_client": elapsed_ms,
                "fallback_used": bool(payload.get("model_fallback_used", False)),
                "is_failure": bool(payload.get("is_failure", False)),
                "low_coverage_warning": bool(payload.get("low_coverage_warning", False)),
                "challenge_risk": float(payload.get("challenge_risk", 0.0)),
                "reliability_grade": payload.get("reliability_grade", "unknown"),
                "keyword_recall": _keyword_recall(answer, item.expected_keywords),
            }
        )

    ok_items = [item for item in results if item.get("ok")]
    metric = lambda key: [float(item[key]) for item in ok_items if key in item and item[key] is not None]
    keyword_recalls = [item["keyword_recall"] for item in ok_items if item.get("keyword_recall") is not None]

    summary = {
        "total_questions": len(results),
        "successful_questions": len(ok_items),
        "failed_questions": len(results) - len(ok_items),
        "avg_confidence": round(statistics.mean(metric("confidence")), 3) if metric("confidence") else 0.0,
        "avg_coverage": round(statistics.mean(metric("coverage")), 3) if metric("coverage") else 0.0,
        "avg_groundedness": round(statistics.mean(metric("groundedness")), 3) if metric("groundedness") else 0.0,
        "avg_latency_ms_server": int(statistics.mean(metric("latency_ms_server"))) if metric("latency_ms_server") else 0,
        "avg_latency_ms_client": int(statistics.mean(metric("latency_ms_client"))) if metric("latency_ms_client") else 0,
        "fallback_rate": round(sum(1 for item in ok_items if item.get("fallback_used")) / max(1, len(ok_items)), 3),
        "failure_rate": round(sum(1 for item in ok_items if item.get("is_failure")) / max(1, len(ok_items)), 3),
        "low_coverage_rate": round(
            sum(1 for item in ok_items if item.get("low_coverage_warning")) / max(1, len(ok_items)),
            3,
        ),
        "avg_keyword_recall": round(statistics.mean(keyword_recalls), 3) if keyword_recalls else None,
        "avg_challenge_risk": round(statistics.mean(metric("challenge_risk")), 3) if metric("challenge_risk") else 0.0,
        "low_reliability_rate": round(
            sum(1 for item in ok_items if item.get("reliability_grade") == "low") / max(1, len(ok_items)),
            3,
        ),
    }

    return {
        "session_id": session_id,
        "settings": {
            "top_k": top_k,
            "max_iterations": max_iterations,
            "fast_mode": fast_mode,
            "deterministic_mode": deterministic_mode,
            "answer_verbosity": answer_verbosity,
            "challenge_mode": challenge_mode,
        },
        "summary": summary,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Recursive Knowledge Engine.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/api")
    parser.add_argument("--queries-file", required=True)
    parser.add_argument("--docs-dir", default=None)
    parser.add_argument("--output", default="./benchmark_report.json")
    parser.add_argument("--session-id", default=f"benchmark-{uuid4()}")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--fast-mode", action="store_true", default=False)
    parser.add_argument("--deterministic-mode", action="store_true", default=False)
    parser.add_argument("--answer-verbosity", default="long", choices=["short", "normal", "long"])
    parser.add_argument("--disable-challenge-audit", action="store_true", default=False)
    parser.add_argument("--upload-batch-size", type=int, default=8)
    args = parser.parse_args()

    queries_file = Path(args.queries_file).resolve()
    output_path = Path(args.output).resolve()
    items = _read_eval_items(queries_file)

    upload_report = None
    if args.docs_dir:
        docs_dir = Path(args.docs_dir).resolve()
        if not docs_dir.exists():
            raise FileNotFoundError(f"Docs dir not found: {docs_dir}")
        upload_report = _upload_docs(args.api_base, docs_dir, batch_size=max(1, args.upload_batch_size))

    report = run_benchmark(
        api_base=args.api_base,
        items=items,
        session_id=args.session_id,
        top_k=max(1, args.top_k),
        max_iterations=max(1, min(3, args.max_iterations)),
        fast_mode=bool(args.fast_mode),
        deterministic_mode=bool(args.deterministic_mode),
        answer_verbosity=args.answer_verbosity,
        challenge_mode=not bool(args.disable_challenge_audit),
    )
    report["upload_report"] = upload_report

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(f"Saved benchmark report: {output_path}")
    print(
        "Summary:",
        f"ok={summary['successful_questions']}/{summary['total_questions']}",
        f"avg_conf={summary['avg_confidence']}",
        f"avg_cov={summary['avg_coverage']}",
        f"avg_ground={summary['avg_groundedness']}",
        f"fallback_rate={summary['fallback_rate']}",
    )


if __name__ == "__main__":
    main()
