from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过正在运行的 FastAPI 服务为评估语料生成 Embedding。"
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="后端服务地址。",
    )
    parser.add_argument(
        "--manifest",
        default="evaluation/corpus_v2/corpus_manifest.json",
        help="语料清单路径。",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="单份文档向量化请求超时秒数。",
    )
    return parser.parse_args()


def request_json(
    url: str,
    method: str = "GET",
    timeout: int = 60,
) -> Any:
    request = urllib.request.Request(
        url=url,
        method=method,
        data=(b"" if method == "POST" else None),
        headers={
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            payload = response.read().decode(
                "utf-8"
            )
            return json.loads(payload) if payload else None

    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            f"{method} {url} failed: "
            f"HTTP {exc.code} {detail}"
        ) from exc


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    manifest = json.loads(
        Path(args.manifest).read_text(
            encoding="utf-8"
        )
    )
    target_filenames = {
        item["filename"]
        for item in manifest["documents"]
    }

    documents = request_json(
        f"{base_url}/documents/",
        timeout=60,
    )
    documents_by_filename = {
        item["filename"]: item
        for item in documents
    }

    missing = sorted(
        target_filenames
        - set(documents_by_filename)
    )
    if missing:
        raise RuntimeError(
            "后端数据库缺少文档: "
            + ", ".join(missing)
        )

    failures: list[str] = []

    for filename in sorted(target_filenames):
        document = documents_by_filename[filename]
        document_id = document["id"]
        status = document["status"]

        if status == "completed":
            print(f"[skip] {filename}: completed")
            continue

        if status not in {
            "chunked",
            "embedding_failed",
        }:
            failures.append(
                f"{filename}: unexpected status={status}"
            )
            continue

        print(
            f"[embedding] {filename} "
            f"(document_id={document_id})"
        )

        try:
            result = request_json(
                (
                    f"{base_url}/documents/"
                    f"{document_id}/embeddings"
                ),
                method="POST",
                timeout=args.timeout,
            )
            print(
                f"[done] {filename}: "
                f"processed_count="
                f"{result.get('processed_count')}"
            )
        except Exception as exc:
            failures.append(
                f"{filename}: {exc}"
            )

    if failures:
        print("\nEmbedding failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)

    print("\nAll evaluation corpus documents are completed.")


if __name__ == "__main__":
    main()
