#!/usr/bin/env python3
"""Turn the WhereOPD multiple-choice counting set into open-ended counting (the 9B data).

    python make_open_ended.py --in whereopd_count_mcq_3k/train.parquet \
                              --out whereopd_count_open_3k/train.parquet

Each row's question "How many <colour> <shapes> are in the image?\\n\\nA. ..\\nB. ..
\\n\\nAnswer with only the letter (A, B, C, or D), nothing else." becomes the bare
question "How many <colour> <shapes> are in the image?" with no options and no
closing instruction, and the ground truth becomes the count as a decimal string
("3") instead of the option letter. Everything else is kept byte-identical: the
image, scene metadata and the where_hint the teacher sees, so the
teacher's privilege is unchanged and only what the student is asked for differs.
"""
import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_scenes import PLURAL  # noqa: E402


def scene(e):
    """(colour, singular shape, [(x, y), ...]) of the target, from the row's scene record."""
    s = e["source_extra_info"]
    return s["target_color"], s["target_shape"], [(int(x), int(y)) for x, y in s["target_coords"]]


def to_open_ended(r):
    e = r["extra_info"]
    color, shape, coords = scene(e)
    question = f"<image>\nHow many {color} {PLURAL[shape]} are in the image?"
    gt = str(len(coords))
    r["prompt"] = [{**t, "content": question} if t.get("role") == "user" else t for t in r["prompt"]]
    e["question"] = question
    e["answer"] = gt
    e["question_type"] = "open"
    r["reward_model"] = {**r["reward_model"], "ground_truth": gt}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    src = pq.ParquetFile(args.src)
    out = Path(args.dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer, n, sample = None, 0, None
    for batch in src.iter_batches(batch_size=args.batch_size):
        rows = [to_open_ended(r) for r in batch.to_pylist()]
        sample = sample or rows[0]
        n += len(rows)
        tbl = pa.Table.from_pylist(rows, schema=src.schema_arrow)
        if writer is None:
            writer = pq.ParquetWriter(out, tbl.schema)
        writer.write_table(tbl)
    if writer:
        writer.close()
    e = sample["extra_info"]
    print(f"wrote {out} ({n} rows)\n  question: {e['question']!r}\n  hint    : {e['where_hint']!r}\n  gt      : {e['answer']!r}")


if __name__ == "__main__":
    main()
