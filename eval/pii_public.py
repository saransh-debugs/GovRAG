"""PII recall on a public third-party dataset: ai4privacy/pii-masking-300k (validation,
English rows). A second opinion on eval/pii_recall.py, whose corpus I wrote myself.

The data is downloaded to ~/.cache/govrag at run time and never committed (its license is
"other"); only aggregate scores go into bench/results/.

Every label is reported, including the ones our default list doesn't cover (passwords,
usernames, street addresses...). Those are the honest gaps; averaging them away would
hide exactly what a reviewer should see.

    python -m eval.pii_public --model en_core_web_md --rows 2000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from govrag.pii.analyzer import PiiAnalyzer
from govrag.pii.entities import DEFAULT_REDACT
from govrag.pii.redact import propagate

URL = ("https://huggingface.co/api/datasets/ai4privacy/pii-masking-300k/parquet/"
       "default/validation/0.parquet")
CACHE = Path.home() / ".cache" / "govrag" / "ai4privacy_val.parquet"

# Their label -> our types that count as correct. Empty = not in our default list.
LABELS: dict[str, tuple[str, ...]] = {
    "GIVENNAME1": ("PERSON",), "GIVENNAME2": ("PERSON",),
    "LASTNAME1": ("PERSON",), "LASTNAME2": ("PERSON",), "LASTNAME3": ("PERSON",),
    "EMAIL": ("EMAIL_ADDRESS",), "TEL": ("PHONE_NUMBER",), "IP": ("IP_ADDRESS",),
    "SOCIALNUMBER": ("NATIONAL_ID", "US_SSN"), "IDCARD": ("NATIONAL_ID",),
    "PASSPORT": ("PASSPORT", "US_PASSPORT"),
    "DRIVERLICENSE": ("DRIVER_LICENSE", "US_DRIVER_LICENSE"),
    "USERNAME": ("USERNAME",), "PASS": ("CREDENTIAL",), "BOD": ("DATE_OF_BIRTH",),
    "STREET": ("STREET_ADDRESS",), "BUILDING": ("STREET_ADDRESS",),
    "SECADDRESS": ("STREET_ADDRESS",),
    # Not default: a tenant opts in with extra_redact (LOCATION, DATE_TIME) or never.
    "CITY": (), "STATE": (), "POSTCODE": (), "COUNTRY": (), "GEOCOORD": (),
    "DATE": (), "TIME": (), "SEX": (), "TITLE": (),
}


def load(rows: int, seed: int) -> list[dict[str, Any]]:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, CACHE)
    import pyarrow.parquet as pq

    en = [r for r in pq.read_table(CACHE, columns=["source_text", "privacy_mask", "language"])
          .to_pylist() if r["language"] == "English"]
    random.Random(seed).shuffle(en)
    return en[:rows]


async def _collect(model: str, rows: int, seed: int) -> dict[str, Any]:
    data = load(rows, seed)
    a = PiiAnalyzer(model=model, ingest_workers=2, query_workers=1)
    await a.start()
    try:
        texts = [r["source_text"] for r in data]
        spans = await a.analyze(texts, DEFAULT_REDACT, lane="ingest", timeout_s=600)
    finally:
        a.close()
    hit: dict[str, int] = defaultdict(int)
    typed: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for r, sp in zip(data, spans, strict=True):
        sp = propagate([r["source_text"]], [sp])[0]
        for m in r["privacy_mask"]:
            lab, gs, ge = m["label"], m["start"], m["end"]
            total[lab] += 1
            cover = [p for p in sp if p[0] <= gs and p[1] >= ge]
            hit[lab] += bool(cover)
            typed[lab] += any(p[2] in LABELS.get(lab, ()) for p in cover)

    def group(keys: list[str]) -> dict[str, float]:
        n = sum(total[k] for k in keys)
        return {"n": n, "hidden_recall": round(sum(hit[k] for k in keys) / max(n, 1), 4)}

    covered = [k for k, v in LABELS.items() if v and total[k]]
    names = [k for k in covered if LABELS[k] == ("PERSON",)]
    return {
        "model": model, "rows": len(data),
        "names": group(names),
        "default_types_all": group(covered),
        "per_label": {k: {"n": total[k], "hidden_recall": round(hit[k] / total[k], 4),
                          "typed_recall": round(typed[k] / total[k], 4),
                          "in_default_list": bool(LABELS.get(k))}
                      for k in sorted(total, key=lambda k: -total[k])},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="en_core_web_md")
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    res = asyncio.run(_collect(a.model, a.rows, a.seed))
    print(json.dumps({k: v for k, v in res.items() if k != "per_label"}, indent=1))
    for k, v in res["per_label"].items():
        print(f"  {k:14} {v}")
    results = Path(__file__).resolve().parent.parent / "bench" / "results"
    out = results / f"pii_public_{a.model}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
