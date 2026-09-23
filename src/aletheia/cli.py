"""命令行入口 (M8).

aletheia ingest <file>       摄入文档
aletheia ask "问题"          推理问答
aletheia serve               启动 HTTP 服务
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .assembly import AletheiaEngine
from .config import default_config


def _read_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader

        r = PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in r.pages)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        prog="aletheia", description="Aletheia 澄明 — 推理时计算 Scaling 引擎")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="摄入文档")
    pi.add_argument("file", help="txt / md / pdf 路径")

    pa = sub.add_parser("ask", help="推理问答")
    pa.add_argument("query")
    pa.add_argument("--strategy", default=None)
    pa.add_argument("--n", type=int, default=None)
    pa.add_argument("--budget", type=float, default=1.0)
    pa.add_argument("--seed", type=int, default=42)

    ps = sub.add_parser("serve", help="启动 HTTP 服务")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)
    engine = AletheiaEngine(default_config())

    if args.cmd == "ingest":
        text = _read_file(args.file)
        print(json.dumps(engine.ingest(text), ensure_ascii=False))
        return 0
    if args.cmd == "ask":
        trace = engine.ask(args.query, args.strategy, args.n, args.budget, args.seed)
        print(f"[strategy] {trace.strategy}")
        print(f"[score]    {trace.final_score:.4f}  ({trace.wall_ms:.1f} ms)")
        print("-" * 60)
        print(trace.answer)
        return 0
    if args.cmd == "serve":
        import uvicorn

        print(f"Aletheia 澄明 服务启动: http://{args.host}:{args.port}")
        uvicorn.run("aletheia.api:app", host=args.host,
                    port=args.port, reload=False, log_level="info")
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
