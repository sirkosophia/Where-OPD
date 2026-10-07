#!/usr/bin/env python3
"""Print the 15-benchmark table (and the average) for evaluated models.

    python eval/summarize.py WhereOPD-Qwen3.5-4B-seed1 WhereOPD-Qwen3.5-4B-seed2 ...

Reads eval/judge/<benchmark>/<model>_seed42_answer.jsonl, written by
scripts/eval_whereopd.sh. A score is the share of items the judge marked
correct. A benchmark with any failed item -- an inference call that errored
("api_error") or a judge call that errored ("judge_error") -- shows ERR instead
of a score, with the counts listed under the table; rerun the evaluation to
retry those items. A benchmark not judged yet shows "-". The average is printed
only when all 15 benchmarks have a score.

Items in EXCLUDED are left out of the score (and the item count) because the
benchmark data itself is broken for them; they are listed under the table.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from benchmarks import SUITE  # noqa: E402
from cal_acc import is_correct  # noqa: E402

ERROR_SOURCES = ("api_error", "judge_error")

# benchmark -> {question_id: reason}. Broken items, excluded for every model.
EXCLUDED = {
    "mme-realworld": {"22438": "image MME_RealWorld_Full_images/022438.jpg is an empty file"},
}


def _qid(r):
    uid = str(r.get("sample_uid") or "")
    return uid.split("question_id:", 1)[1] if uid.startswith("question_id:") else str(r.get("question_id"))


def score(path: Path, bench: str):
    """(accuracy or None, {error source: count}, n items); (None, {}, 0) if not judged."""
    if not path.exists():
        return None, {}, 0
    skip = EXCLUDED.get(bench, {})
    records = [r for r in json.load(open(path, encoding="utf-8")) if _qid(r) not in skip]
    if not records:
        return None, {}, 0
    errors = {}
    for r in records:
        src = r.get("judge_source")
        if src in ERROR_SOURCES:
            errors[src] = errors.get(src, 0) + 1
    if errors:
        return None, errors, len(records)
    return 100.0 * sum(is_correct(r) for r in records) / len(records), {}, len(records)


def main():
    models = sys.argv[1:]
    if not models:
        sys.exit(__doc__)
    width = max(12, max(len(m) for m in models))
    print(f"{'model':{width}s}" + "".join(f"{disp[:8]:>9s}" for _, _, disp in SUITE) + "      avg")
    notes = []
    for m in models:
        cells, vals = [], []
        for name, _, disp in SUITE:
            acc, errors, n = score(HERE / "judge" / name / f"{m}_seed42_answer.jsonl", name)
            vals.append(acc)
            if errors:
                cells.append(f"{'ERR':>9s}")
                detail = ", ".join(f"{c} {src}" for src, c in errors.items())
                notes.append(f"  {m} / {disp}: {detail} of {n} items")
            elif acc is None:
                cells.append(f"{'-':>9s}")
            else:
                cells.append(f"{acc:9.2f}")
        avg = f"{sum(vals) / len(vals):9.2f}" if all(v is not None for v in vals) else f"{'-':>9s}"
        print(f"{m:{width}s}{''.join(cells)}{avg}")
    disp = {name: d for name, _, d in SUITE}
    print("\nExcluded from the score (broken benchmark data, same for every model):")
    for bench, items in EXCLUDED.items():
        for qid, why in items.items():
            print(f"  {disp.get(bench, bench)} question {qid}: {why}")
    if notes:
        print("\nERR = items that failed (api_error: inference call failed; judge_error: judge call failed).")
        print("Rerun the evaluation for these; it retries only the failed items:")
        print("\n".join(notes))


if __name__ == "__main__":
    main()
