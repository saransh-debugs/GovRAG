"""PII detection recall/precision on the synthetic corpus, with and without propagation.

hidden recall: the gold value is fully covered by some detected span (any type).
               A miss here = raw PII reaches the LLM. This is the leak number.
typed recall:  covered by a span of the *right* type. A miss here = the value is hidden
               but under the wrong type, so the wrong unmask rule applies to it.
precision:     detected spans that overlap some gold span. Low precision = over-redaction
               (annoying, not a leak).

    python -m eval.pii_recall
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

from eval.synth_corpus import Doc, Span, build
from govrag.config import get_settings
from govrag.pii.analyzer import PiiAnalyzer
from govrag.pii.analyzer import Span as Pred
from govrag.pii.entities import DEFAULT_REDACT
from govrag.pii.redact import propagate


def score(docs: list[Doc], preds: list[list[list[Pred]]]) -> dict[str, dict[str, float]]:
    hit: dict[str, int] = defaultdict(int)
    typed: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    tp_pred = 0
    n_pred = 0
    for doc, dpreds in zip(docs, preds, strict=True):
        for gold, ps in zip(doc.gold, dpreds, strict=True):
            for gs, ge, gt in gold:
                total[gt] += 1
                cover = [p for p in ps if p[0] <= gs and p[1] >= ge]
                if cover:
                    hit[gt] += 1
                if any(p[2] == gt for p in cover):
                    typed[gt] += 1
            n_pred += len(ps)
            tp_pred += sum(1 for p in ps if any(p[0] < ge and p[1] > gs for gs, ge, _ in gold))
    out = {t: {"n": total[t], "hidden_recall": round(hit[t] / total[t], 4),
               "typed_recall": round(typed[t] / total[t], 4)} for t in sorted(total)}
    all_n = sum(total.values())
    out["ALL"] = {"n": all_n, "hidden_recall": round(sum(hit.values()) / all_n, 4),
                  "typed_recall": round(sum(typed.values()) / all_n, 4),
                  "precision": round(tp_pred / max(n_pred, 1), 4)}
    return out


def _leaks(docs: list[Doc], preds: list[list[list[Pred]]], limit: int = 8) -> list[str]:
    out = []
    for doc, dpreds in zip(docs, preds, strict=True):
        for text, gold, ps in zip(doc.sections, doc.gold, dpreds, strict=True):
            for gs, ge, gt in gold:
                if not any(p[0] <= gs and p[1] >= ge for p in ps):
                    # Show the context, not just the value: the pattern is what matters.
                    out.append(f"{gt}: ...{text[max(0, gs - 25):ge + 25]}...")
                    if len(out) >= limit:
                        return out
    return out


async def _collect() -> dict[str, object]:
    s = get_settings()
    docs = build()
    a = PiiAnalyzer(model=s.spacy_model, ingest_workers=2, query_workers=1)
    await a.start()
    try:
        t0 = time.perf_counter()
        raw = [await a.analyze(d.sections, DEFAULT_REDACT, lane="ingest") for d in docs]
        took = time.perf_counter() - t0
    finally:
        a.close()
    prop = [propagate(d.sections, r) for d, r in zip(docs, raw, strict=True)]
    result = {
        "docs": len(docs),
        "sections": sum(len(d.sections) for d in docs),
        "analyze_seconds": round(took, 1),
        "presidio_only": score(docs, raw),
        "with_propagation": score(docs, prop),
        "sample_misses_with_propagation": _leaks(docs, prop),
    }
    for k in ("presidio_only", "with_propagation"):
        print(k)
        for t, v in result[k].items():  # type: ignore[union-attr]
            print(f"  {t:14} {v}")
    print("sample misses:", *result["sample_misses_with_propagation"], sep="\n  ")  # type: ignore[arg-type]
    return result


def main() -> None:
    result = asyncio.run(_collect())
    out = Path(__file__).resolve().parent.parent / "bench" / "results" / "pii_recall.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()


__all__ = ["Span", "score"]
