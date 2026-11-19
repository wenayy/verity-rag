"""Command line: ``verity ingest | ask | chat | eval | serve``."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .config import get_settings


def cmd_ingest(args) -> None:
    from .index.store import KnowledgeIndex

    s = get_settings()
    idx = KnowledgeIndex.from_settings(s)
    idx.save(s.index_dir)
    print(f"indexed {len(idx.doc_titles)} documents into {len(idx.chunks)} chunks "
          f"({s.chunk_strategy}, {idx.embedder.name} embeddings) -> {s.index_dir}")


def _print_answer(a) -> None:
    print(f"\n{a.answer}\n")
    for c in a.citations:
        print(f"  [{c.n}] {c.title}" + (f" › {c.section.split(' > ')[-1]}" if c.section else ""))
    from .tracing import STORE

    t = STORE.get(a.trace_id)
    if t:
        print(f"\n  {t.root.duration_ms:.0f} ms · {t.usage.llm_calls} LLM calls · "
              f"{t.usage.prompt_tokens + t.usage.completion_tokens} tokens · ${t.usage.cost_usd:.4f} · "
              f"{a.iterations} retrieval round(s) · grounded={a.grounded}")


def cmd_ask(args) -> None:
    from .agent.agent import build_agent

    agent = build_agent()
    for ev in agent.stream(args.question):
        if ev.type == "step" and args.verbose:
            print(f"  · {json.dumps(ev.data)[:200]}", file=sys.stderr)
        elif ev.type == "final":
            _print_answer(ev.data["answer"])


def cmd_chat(args) -> None:
    from .agent.agent import build_agent

    agent, history = build_agent(), []
    print("Verity · Helix Freight demo knowledge base. Ctrl-D to exit.")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            continue
        a = agent.ask(q, history)
        _print_answer(a)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": a.answer}]


def cmd_eval(args) -> None:
    from .eval import runner
    from .ingest.loaders import load_documents

    s = get_settings()
    golden = runner.load_golden(Path(args.golden))
    if args.limit:
        golden = golden[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode in ("retrieval", "all"):
        llm = None
        if args.with_llm:
            from .llm import OpenAICompatLLM

            llm = OpenAICompatLLM(s.llm_model, s.llm_base_url, s.llm_api_key)
        docs = load_documents(s.kb_dir)
        subsets = {"all questions": golden,
                   "hard (paraphrase + multi-hop)": [g for g in golden if g.category in ("paraphrase", "multi_hop")]}
        md = [f"# Retrieval evaluation (k={args.k})\n"]
        payload = {}
        for label, items in subsets.items():
            reps = runner.run_ablation(docs, items, k=args.k, llm=llm)
            md += [f"## {label} (n={len(items)})\n", runner.ablation_markdown(reps, args.k), ""]
            best = max(reps, key=lambda r: (r.mrr, r.ndcg))
            md += [f"Per-category breakdown for best config ({best.name}):\n", runner.category_markdown(best), ""]
            payload[label] = [asdict(r) for r in reps]
        (out / "retrieval.md").write_text("\n".join(md))
        (out / "retrieval.json").write_text(json.dumps(payload, indent=2))
        print("\n".join(md))

    if args.mode in ("answers", "all"):
        from .agent.agent import build_agent
        from .llm import OpenAICompatLLM

        judge_llm = OpenAICompatLLM(args.judge_model or s.llm_model, s.llm_base_url, s.llm_api_key)
        res = runner.eval_answers(build_agent(s), judge_llm, golden, k=args.k)
        (out / "answers.json").write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps(res["summary"], indent=2))


def cmd_serve(args) -> None:
    import os

    import uvicorn

    if args.demo:
        os.environ["VERITY_DEMO"] = "1"
    uvicorn.run("verity.api.server:create_app", factory=True, host=args.host, port=args.port, reload=False)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="verity", description="Verity: self-verifying agentic RAG")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="chunk, embed and index the knowledge base").set_defaults(fn=cmd_ingest)
    a = sub.add_parser("ask", help="ask one question")
    a.add_argument("question")
    a.add_argument("-v", "--verbose", action="store_true", help="print agent steps")
    a.set_defaults(fn=cmd_ask)
    sub.add_parser("chat", help="interactive multi-turn chat").set_defaults(fn=cmd_chat)
    e = sub.add_parser("eval", help="run the evaluation harness")
    e.add_argument("--mode", choices=["retrieval", "answers", "all"], default="retrieval")
    e.add_argument("--golden", default="evals/golden.jsonl")
    e.add_argument("--out", default="evals/results")
    e.add_argument("-k", type=int, default=5)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--with-llm", action="store_true", help="also ablate LLM query expansion and reranking")
    e.add_argument("--judge-model", default=None)
    e.set_defaults(fn=cmd_eval)
    sv = sub.add_parser("serve", help="run the API + web UI")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--demo", action="store_true", help="offline extractive model, no API key needed")
    sv.set_defaults(fn=cmd_serve)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
