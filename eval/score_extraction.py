"""Score extraction output (output/<bid>.json) against eval/extraction_expected.json.

Per field (see the file's "_about"):
  1.0  correct  - every key-term group present, cited to an expected source file
  0.5  partial  - at least half of the groups present and correctly cited (incomplete value),
                  or all present but `full_credit_requires` (e.g. a named person) missing
  0.0  wrong / missing / invented (value where null is expected) / unsupported (no source
       from an expected file)
Key terms are compared after Unicode normalisation (no-break / thin spaces, dash variants) and
only count when stated positively: "warranty not found in provided passages" does not earn
the warranty group. A citation or a high confidence earns nothing on its own. Accuracy = mean score over the
20 fields; "strict" = share of fields scored 1.0. Addendum changes are checked separately.

    python -m eval.score_extraction [--out output] [--bids Bid1 Bid2]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extraction.textnorm import normalize_text, states_positively  # noqa: E402

HERE = Path(__file__).resolve().parent


def _text(value) -> str:
    return " ".join(value) if isinstance(value, list) else str(value or "")


def _has(text: str, alts: list[str]) -> bool:
    """Plain containment after Unicode normalisation ("16 GB" == "16 GB")."""
    t = normalize_text(text)
    return any(normalize_text(a) in t for a in alts)


def _states(text: str, alts: list[str]) -> bool:
    """Like `_has`, but a negated or hedged mention ("warranty not found in provided passages")
    earns nothing: one alternative must be stated positively."""
    return any(states_positively(re.escape(normalize_text(a)), text) for a in alts)


def score_field(exp: dict, got: dict) -> tuple[float, str]:
    value, sources = got.get("value"), got.get("sources") or []
    text = _text(value)
    if exp["expect"] == "null":
        if value is None:
            return 1.0, "correct: null (absent in documents)"
        if exp.get("null_values") and _has(text, exp["null_values"]):
            return 1.0, "correct: states absence / equivalent"
        return 0.0, f"invented: value where the documents state nothing ({text[:60]!r})"
    if value is None:
        return 0.0, "missing: null although the documents state it"
    files = [s.get("file", "") for s in sources]
    cited_ok = any(any(e.lower() in f.lower() for e in exp.get("sources", [""])) for f in files)
    if not cited_ok:
        return 0.0, f"unsupported: no source from an expected file (cited {files[:3]})"
    groups = exp.get("all_of", [])
    hit = sum(_states(text, g) for g in groups)
    missing = [g[0] for g in groups if not _states(text, g)]
    if hit == len(groups):
        extra = [g[0] for g in exp.get("full_credit_requires", []) if not _states(text, g)]
        if extra:
            return 0.5, f"partial: missing {extra}"
        return 1.0, "correct"
    if groups and hit / len(groups) >= 0.5:
        return 0.5, f"partial: missing {missing}"
    return 0.0, f"incorrect: missing {missing}"


def score_bid(bid: str, expected: dict, record: dict) -> dict:
    rows = []
    for name, exp in expected[bid].items():
        got = record["fields"].get(name, {})
        score, why = score_field(exp, got)
        rows.append({"field": name, "score": score, "verdict": why, "status": got.get("status"),
                     "confidence": got.get("confidence"), "value": _text(got.get("value"))[:160] or None,
                     "sources": sorted({f"{s['file']} p.{s.get('page')}" for s in got.get("sources") or []})})
    # addendum changes
    exp_changes = expected.get("addendum_changes", {}).get(bid, [])
    got_changes = record.get("addendum_changes", [])
    change_rows = []
    for ec in exp_changes:
        gc = next((c for c in got_changes if c["field"] == ec["field"]), None)
        ok = bool(gc) and _has(_text(gc.get("previous_value")), ec["previous_any"]) and \
            _has(_text(gc.get("value")), ec["value_any"]) and gc.get("addendum_number") == ec.get("addendum_number")
        change_rows.append({"field": ec["field"], "expected": True, "found": bool(gc), "correct": ok})
    for gc in got_changes:
        if not any(ec["field"] == gc["field"] for ec in exp_changes):
            change_rows.append({"field": gc["field"], "expected": False, "found": True, "correct": False,
                                "note": f"unexpected change: {_text(gc.get('previous_value'))[:60]!r} -> "
                                        f"{_text(gc.get('value'))[:60]!r}"})
    total = sum(r["score"] for r in rows)
    return {"bid_id": bid, "score": total, "accuracy": round(total / len(rows), 3),
            "strict_accuracy": round(sum(r["score"] == 1.0 for r in rows) / len(rows), 3),
            "fields": rows, "addendum_changes": change_rows, "model": record.get("model"),
            "llm_calls": record.get("llm_calls"), "validation": record.get("validation")}


def to_markdown(results: list[dict]) -> str:
    out = ["# Extraction accuracy", ""]
    for r in results:
        out += [f"## {r['bid_id']} - accuracy {r['accuracy']:.1%} (strict {r['strict_accuracy']:.1%}), "
                f"score {r['score']}/20, model {r['model']}, {r['llm_calls']} LLM calls", "",
                "| Field | Score | Verdict | Status | Conf. | Value | Sources |", "|---|---|---|---|---|---|---|"]
        for f in r["fields"]:
            val = (f["value"] or "–").replace("|", "\\|")
            out.append(f"| {f['field']} | {f['score']} | {f['verdict']} | {f['status']} | {f['confidence']} | "
                       f"{val} | {'; '.join(f['sources'])[:120]} |")
        out += ["", "Addendum changes:", ""]
        out += [f"* {c['field']}: " + ("correct" if c["correct"] else c.get("note", "wrong or missing"))
                for c in r["addendum_changes"]] or ["* none expected, none recorded"]
        out.append("")
    if len(results) > 1:
        tot = sum(r["score"] for r in results) / (20 * len(results))
        out.append(f"**Overall accuracy: {tot:.1%}** over {20 * len(results)} fields.")
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=HERE.parent / "output")
    p.add_argument("--bids", nargs="+", default=["Bid1", "Bid2"])
    args = p.parse_args()
    expected = json.loads((HERE / "extraction_expected.json").read_text(encoding="utf-8"))
    results = []
    for bid in args.bids:
        path = args.out / f"{bid}.json"
        if not path.exists():
            print(f"{bid}: no output at {path} (not run yet)")
            continue
        results.append(score_bid(bid, expected, json.loads(path.read_text(encoding="utf-8"))))
    if not results:
        return
    res_dir = HERE / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "extraction_accuracy.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = to_markdown(results)
    (res_dir / "extraction_accuracy.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
