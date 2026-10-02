import argparse
import base64
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

from huggingface_hub import snapshot_download

# The 15 benchmarks of the WhereOPD evaluation suite.
BENCHMARK_JSON_MAP = {
    "cvbench": "cvbench.json",
    "mme-realworld": "MME_RealWorld.json",
    "blink": "blink_full.json",
    "gqa": "gqa.json",
    "countqa": "countqa.json",
    "chartqa": "chartqa.json",
    "docvqa": "docvqa.json",
    "ocrbench": "ocrbench.json",
    "evochart": "evochart.json",
    "hallusionbench": "hallusionbench.json",
    "amber": "amber.json",
    "vstar": "vstar.json",
    "hrbench-4k": "hr_bench_4k.json",
    "hrbench-8k": "hr_bench_8k.json",
    "zoombench": "zoombench.json",
}


def resolve_benchmark_json(benchmark):
    if benchmark not in BENCHMARK_JSON_MAP:
        print(f"ERROR: Unsupported benchmark: {benchmark}", file=sys.stderr)
        print(f"Supported: {', '.join(BENCHMARK_JSON_MAP.keys())}", file=sys.stderr)
        sys.exit(1)
    return BENCHMARK_JSON_MAP[benchmark]


def _to_pil(img):
    """Accept a PIL image, raw bytes, or HF's {"bytes": ..., "path": ...} dict.

    load_dataset(streaming=True) hands back the dict form rather than a decoded
    PIL object, which crashed the CLEVR and PlotQA builders on their first run.
    """
    if isinstance(img, dict):
        img = img.get("bytes") or img.get("path")
    if isinstance(img, (bytes, bytearray)):
        from io import BytesIO
        from PIL import Image
        return Image.open(BytesIO(img))
    if isinstance(img, str):
        from PIL import Image
        return Image.open(img)
    return img


def _save_image(img, path):
    """Write a PIL image once; skip if already on disk (idempotent rebuilds)."""
    if path.exists():
        return
    img = _to_pil(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(path, "JPEG", quality=95)


def prepare_evochart(out_dir, split="train"):
    """EvoChart-QA: 1250 chart question-answering items.

    Read from gsarch/EvoChart-QA rather than the official
    MuyeHuang/EvoChart-QA-Benchmark, which ships a zip with no parquet and
    exposes only an `image` column through the datasets server.

    Answers are short values ("32", "25.5", a label), so this is free-form, not
    multiple choice -- it stays out of judge_qwenlm.py's MCQ_BENCHMARKS and
    grades through grade_answer() (mathruler handles numeric equivalence, which
    matters for chart reads) plus the generic LLM judge.

    The single-word/phrase instruction is the standard chart-QA prompt; without
    it models narrate the chart and the value has to be dug out of prose.
    """
    from datasets import load_dataset

    ds = load_dataset("gsarch/EvoChart-QA", split=split)
    img_dir = out_dir / "EvoChart_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    suffix = "\nAnswer the question using a single word or phrase."
    records = []
    for idx, row in enumerate(tqdm(ds, desc="evochart")):
        img = row.get("image")
        if img is None:
            continue
        img_path = img_dir / f"evochart_{idx:05d}.jpg"
        _save_image(img, img_path)
        q = (row.get("question") or "").strip()
        a = str(row.get("answer") or "").strip()
        if not q or not a:
            continue
        records.append({
            "images": [str(img_path)],
            "query": q + suffix,
            "response": a,
            "question_id": idx,
            "attribute": row.get("attribute") or "",
            "chart_type": row.get("chart_type") or "",
            "is_clear": bool(row.get("is_clear", False)),
        })
    return records


def prepare_countqa(out_dir):
    """CountQA (arXiv:2508.06585): open-ended integer counting.

    1001 hand-captured images, each carrying one or more "How many X?" questions
    in parallel `questions`/`answers` lists. Flattened here to one record per QA
    pair (1528 total), which is what infer.py expects; the image is written once
    and shared by the records referencing it.

    Not multiple choice -- deliberately absent from judge_qwenlm.py's
    MCQ_BENCHMARKS, so it grades via grade_answer() plus the generic
    "same meaning" LLM judge, like zoombench.
    """
    from datasets import load_dataset

    ds = load_dataset("Jayant-Sravan/CountQA", split="test")
    img_dir = out_dir / "CountQA_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # A terse format nudge keeps free-text answers parseable without changing the
    # task, mirroring the "Select from the following choices." the MCQ sets carry.
    suffix = "\nAnswer with a single number."

    records, qid = [], 0
    for idx, row in enumerate(tqdm(ds, desc="countqa")):
        img_path = img_dir / f"countqa_{idx:05d}.jpg"
        if not img_path.exists():
            img = row["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(img_path, "JPEG", quality=95)

        questions = row.get("questions") or []
        answers = row.get("answers") or []
        if len(questions) != len(answers):
            continue
        objects = row.get("objects") or []
        categories = row.get("categories") or []

        for q, a in zip(questions, answers):
            q, a = (q or "").strip(), (a or "").strip()
            if not q or not a:
                continue
            records.append({
                "images": [str(img_path)],
                "query": q + suffix,
                "response": a,
                "question_id": qid,
                "image_id": idx,
                "objects": [o.strip() for o in objects],
                "category": categories[0].strip() if categories else "",
                "is_focused": bool(row.get("is_focused", False)),
            })
            qid += 1

    return records


def prepare_zoombench(out_dir):
    import pyarrow.parquet as pq

    local_dir = out_dir / "ZoomBench_data"
    snapshot_download("inclusionAI/ZoomBench", repo_type="dataset", local_dir=str(local_dir))

    src = local_dir / "data" / "test.parquet"
    full_dir = out_dir / "ZoomBench_images"
    crop_dir = out_dir / "ZoomBench_crop_images"
    full_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(str(src), columns=["id", "query", "response", "image", "crop_image"])
    rows = table.to_pylist()

    data = []
    for i, row in enumerate(rows):
        sid = str(row.get("id") or i)
        full_obj = row.get("image") or {}
        full_bytes = full_obj.get("bytes") if isinstance(full_obj, dict) else None
        if not isinstance(full_bytes, (bytes, bytearray)):
            raise ValueError(f"Invalid full image bytes at row {i}")

        crop_obj = row.get("crop_image") or {}
        crop_bytes = crop_obj.get("bytes") if isinstance(crop_obj, dict) else None

        full_path = full_dir / f"{sid}.jpg"
        with open(full_path, "wb") as f:
            f.write(full_bytes)

        crop_list = []
        if isinstance(crop_bytes, (bytes, bytearray)):
            crop_path = crop_dir / f"{sid}.jpg"
            with open(crop_path, "wb") as f:
                f.write(crop_bytes)
            crop_list = [str(crop_path)]

        data.append({
            "images": [str(full_path)],
            "crop_images": crop_list,
            "query": (row.get("query") or "").strip(),
            "response": (row.get("response") or "").strip(),
        })
    return data


def prepare_vstar(out_dir):
    import pyarrow.parquet as pq

    local_dir = out_dir / "vstar_data"
    snapshot_download("lmms-lab/vstar-bench", repo_type="dataset", local_dir=str(local_dir))

    src = local_dir / "data" / "test-00000-of-00001.parquet"
    img_dir = out_dir / "VStar_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(str(src), columns=["image", "text", "label", "question_id", "category"])
    rows = table.to_pylist()

    data = []
    for i, row in enumerate(rows):
        qid = row.get("question_id")
        if qid is None or str(qid).strip() == "":
            qid = i
        image_obj = row.get("image") or {}
        img_bytes = image_obj.get("bytes") if isinstance(image_obj, dict) else None
        if not isinstance(img_bytes, (bytes, bytearray)):
            raise ValueError(f"Invalid image bytes at row {i}, question_id={qid}")

        img_path = img_dir / f"{str(qid)}.jpg"
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        query_raw = (row.get("text") or "").strip()
        query = query_raw
        post_prompt = "\nAnswer with the option's letter from the given choices directly."
        if not query.endswith(post_prompt.strip()):
            query += post_prompt
        response = (row.get("label") or "").strip().upper()
        data.append({
            "images": [str(img_path)],
            "query": query,
            "response": response,
            "question_id": qid,
            "category": row.get("category") or "unknown",
        })
    return data


def prepare_hrbench(out_dir, benchmark):
    import pyarrow.parquet as pq

    local_dir = out_dir / "HR-Bench_data"
    snapshot_download("DreamMr/HR-Bench", repo_type="dataset", local_dir=str(local_dir))

    parquet_name = "hr_bench_4k.parquet" if benchmark == "hrbench-4k" else "hr_bench_8k.parquet"
    src = local_dir / parquet_name
    img_dir = out_dir / ("HRBench_4k_images" if benchmark == "hrbench-4k" else "HRBench_8k_images")
    img_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(str(src), columns=["index", "question", "answer", "A", "B", "C", "D", "category", "image"])
    rows = table.to_pylist()

    data = []
    for i, row in enumerate(rows):
        idx = int(row.get("index", i))
        img_path = img_dir / f"{idx:05d}.jpg"
        img_b64 = row.get("image") or ""
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img_b64))
        question = (row.get("question") or "").strip()
        options = []
        for letter in ["A", "B", "C", "D"]:
            opt = (row.get(letter) or "").strip()
            if opt:
                options.append(f"({letter}) {opt}")
        if options:
            query = question + " Select from the following choices.\n" + "\n".join(options)
        else:
            query = question
        data.append({
            "index": idx,
            "question_id": idx,
            "images": [str(img_path)],
            "query": query,
            "response": (row.get("answer") or "").strip(),
            "category": (row.get("category") or "").strip(),
        })
    return data


def _load_mme_realworld_parquet(repo_id, out_dir, img_subdir):
    import pyarrow.dataset as ds

    local_dir = out_dir / repo_id.replace("/", "_")
    snapshot_download(repo_id, repo_type="dataset", local_dir=str(local_dir))

    data_dir = local_dir / "data"
    img_dir = out_dir / img_subdir
    img_dir.mkdir(parents=True, exist_ok=True)

    dataset = ds.dataset(str(data_dir), format="parquet")
    table = dataset.to_table(columns=["bytes", "index", "question", "multi-choice options", "answer", "category", "l2-category"])
    return table.to_pylist(), img_dir


def _build_mme_realworld_query(question, option_lines, lang="en"):
    if lang == "cn":
        return (
            question
            + " 选项如下所示:\n"
            + "\n".join(option_lines)
            + "\n根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n"
            + "最佳答案为： "
        ) if option_lines else question
    return (
        question
        + " The choices are listed below:\n"
        + "\n".join(option_lines)
        + "\nSelect the best answer to the above multiple-choice question based on the image. "
        + "Respond with only the letter (A, B, C, D, or E) of the correct option.\n"
        + "The best answer is: "
    ) if option_lines else question


def _process_mme_realworld_rows(rows, img_dir, lang="en"):
    data = []
    for row in tqdm(rows, desc="Processing images", unit="img"):
        idx = int(row.get("index", 0))
        b64 = row.get("bytes") or ""
        img_path = img_dir / f"{idx:06d}.jpg"
        if not img_path.exists():
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(b64))

        question = (row.get("question") or "").strip()
        options = row.get("multi-choice options") or []
        option_lines = [str(x).strip() for x in options if str(x).strip()]
        query = _build_mme_realworld_query(question, option_lines, lang)
        answer = (row.get("answer") or "").strip()

        data.append({
            "index": idx,
            "question_id": idx,
            "images": [str(img_path)],
            "query": query,
            "response": answer,
        })
    return data


def prepare_mme_realworld(out_dir):
    rows, img_dir = _load_mme_realworld_parquet(
        "yifanzhang114/MME-RealWorld-Lmms-eval", out_dir, "MME_RealWorld_Full_images",
    )
    return _process_mme_realworld_rows(rows, img_dir, lang="en")


def _prepare_cvbench_rows(out_dir, task_filter=None):
    import pyarrow.parquet as pq
    import re

    local_dir = out_dir / "CVBench_data"
    snapshot_download("nyu-visionx/CV-Bench", repo_type="dataset", local_dir=str(local_dir))

    img_dir = out_dir / "CVBench_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(local_dir.glob("**/*.parquet"))

    data = []
    idx = 0
    for pf in parquet_files:
        table = pq.read_table(str(pf))
        rows = table.to_pylist()
        for row in tqdm(rows, desc=f"Processing {pf.name}", unit="row"):
            task = str(row.get("task") or "").strip()
            if task_filter and task.lower() != task_filter.lower():
                continue

            image_obj = row.get("image") or {}
            img_bytes = image_obj.get("bytes") if isinstance(image_obj, dict) else None
            if not isinstance(img_bytes, (bytes, bytearray)):
                continue

            img_path = img_dir / f"cvbench_{idx:05d}.jpg"
            with open(img_path, "wb") as f:
                f.write(img_bytes)

            query = (row.get("prompt") or "").strip()
            if not query:
                question = (row.get("question") or "").strip()
                choices = [str(c).strip() for c in (row.get("choices") or [])]
                opts = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices) if i < 4)
                query = f"{question} Select from the following choices.\n{opts}"

            raw_answer = (row.get("answer") or "").strip().upper()
            m = re.search(r"([A-D])", raw_answer)
            answer = m.group(1) if m else raw_answer

            data.append({
                "images": [str(img_path)],
                "query": query,
                "response": answer,
                "question_id": row.get("idx", idx),
                "task": task,
                "source": str(row.get("source") or ""),
            })
            idx += 1

    return data


def prepare_cvbench(out_dir):
    return _prepare_cvbench_rows(out_dir, task_filter=None)


BLINK_SUBTASKS = [
    "Art_Style", "Counting", "Forensic_Detection", "Functional_Correspondence",
    "IQ_Test", "Jigsaw", "Multi-view_Reasoning", "Object_Localization",
    "Relative_Depth", "Relative_Reflectance", "Semantic_Correspondence",
    "Spatial_Relation", "Visual_Correspondence", "Visual_Similarity",
]


def prepare_blink_full(out_dir):
    """Full BLINK (Fu et al. 2024, arXiv:2404.12390), val split: all 14
    subtasks, 1,901 items, letter-MCQ throughout.

    MULTI-IMAGE. Jigsaw and Visual_Similarity carry three images, the
    correspondence and multi-view tasks two, and their prompts refer to them
    positionally ("the first image", "the second image"). Every present image
    goes into records["images"] IN ORDER, and infer.py sends one image_url
    block per entry -- feeding only the first image would leave roughly a third
    of the benchmark unanswerable while still emitting confident letters, so
    the scores would look plausible and mean nothing.

    Uses BLINK's own `prompt` field rather than rebuilding question+choices:
    it is the exact string the authors evaluate with, including the
    positional image wording that our reconstruction would lose.

    `sub_task` is kept on each record so the 14 subtasks can be scored
    separately -- the headline BLINK number is the mean over subtasks, and
    Counting is one of the 14 (our existing blink-counting benchmark is the
    same 120 items, so the two must agree).
    """
    from datasets import load_dataset

    # Absolute, like every other builder here: a relative --data_dir would
    # otherwise bake cwd-dependent paths into the JSON, which resolve only when
    # the harness happens to run from the repo root.
    img_dir = (out_dir / "BLINK_images").resolve()
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    idx = 0
    for sub in BLINK_SUBTASKS:
        ds = load_dataset("BLINK-Benchmark/BLINK", sub, split="val")
        for row in tqdm(ds, desc=f"blink:{sub}"):
            paths = []
            for slot in ("image_1", "image_2", "image_3", "image_4"):
                img = row.get(slot)
                if img is None:
                    continue
                if img.mode != "RGB":
                    img = img.convert("RGB")
                p = img_dir / f"blink_{idx:05d}_{slot}.jpg"
                if not p.exists():
                    img.save(p, "JPEG", quality=95)
                paths.append(str(p))
            if not paths:
                continue
            ans = (row.get("answer") or "").strip("() ").upper()
            if not ans:
                continue
            records.append({
                "images": paths,
                "query": row["prompt"],
                "response": ans,
                "question_id": idx,
                "image_id": idx,
                "sub_task": sub,
                "blink_idx": row.get("idx"),
            })
            idx += 1
    return records


SHORT_ANSWER_SUFFIX = "\nAnswer the question using a single word or phrase."


def _prepare_docvqa_family(out_dir, config, img_subdir, tag):
    """DocVQA / InfographicVQA (Mathew et al.), validation split.

    The test splits are leaderboard-only (no public answers), so validation is
    the evaluable set: DocVQA 5,349 items, InfographicVQA 2,801.

    Each item carries a LIST of acceptable answers (DocVQA median 2, max 5) --
    annotator variants such as "0.28" / "0.28 per 1000". The official metric is
    ANLS, scored as the MAX over those variants, so the full list is kept;
    collapsing to answers[0] would mark correct responses wrong.

    Images are large scans (DocVQA runs to ~2257x1764) and many are grayscale
    mode "L", hence the RGB conversion.
    """
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/DocVQA", config, split="validation")
    img_dir = (out_dir / img_subdir).resolve()
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, row in enumerate(tqdm(ds, desc=tag)):
        img = row.get("image")
        if img is None:
            continue
        answers = [str(a).strip() for a in (row.get("answers") or []) if str(a).strip()]
        question = (row.get("question") or "").strip()
        if not answers or not question:
            continue
        img_path = img_dir / f"{tag}_{idx:05d}.jpg"
        _save_image(img, img_path)
        records.append({
            "images": [str(img_path)],
            "query": question + SHORT_ANSWER_SUFFIX,
            # `response` is the first variant so anything reading a single
            # string still works; `answers` is what the ANLS scorer uses.
            "response": answers[0],
            "answers": answers,
            "question_id": idx,
            "image_id": idx,
            "source_question_id": str(row.get("questionId") or ""),
        })
    return records


def prepare_docvqa(out_dir):
    return _prepare_docvqa_family(out_dir, "DocVQA", "DocVQA_images", "docvqa")


def prepare_gqa(out_dir):
    """GQA testdev-balanced (Hudson & Manning 2019): 12,578 questions.

    testdev_balanced is the split the literature reports. Questions and images
    live in SEPARATE configs on the hub and must be joined on imageId -- there
    are only 398 distinct images for 12,578 questions, so each image is written
    once and referenced by the ~32 questions that share it. Writing one copy per
    question would be 12,578 files for no reason.

    Single short answer ("no", "left", "shirt"), graded by normalized exact
    match, which is GQA's own metric -- no LLM judge.
    """
    from datasets import load_dataset

    img_dir = (out_dir / "GQA_images").resolve()
    img_dir.mkdir(parents=True, exist_ok=True)

    imgs = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")
    id_to_path = {}
    for row in tqdm(imgs, desc="gqa:images"):
        iid = str(row["id"])
        p = img_dir / f"gqa_{iid}.jpg"
        _save_image(row["image"], p)
        id_to_path[iid] = str(p)

    qs = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
    records, missing = [], 0
    for idx, row in enumerate(tqdm(qs, desc="gqa:questions")):
        p = id_to_path.get(str(row.get("imageId")))
        if p is None:
            missing += 1
            continue
        q = (row.get("question") or "").strip()
        a = str(row.get("answer") or "").strip()
        if not q or not a:
            continue
        records.append({
            "images": [p],
            "query": q + SHORT_ANSWER_SUFFIX,
            "response": a,
            "question_id": idx,
            "source_question_id": str(row.get("id") or ""),
            "image_id": str(row.get("imageId") or ""),
        })
    if missing:
        print(f"WARNING: {missing} questions had no matching image", file=sys.stderr)
    return records


def prepare_ocrbench(out_dir):
    """OCRBench (echo840/OCRBench): 1,000 items over 10 OCR task types.

    50 each of Regular/Irregular/Artistic/Handwriting/Digit-String/Non-Semantic
    text recognition, 200 each of Scene Text-centric VQA, Doc-oriented VQA and
    Key Information Extraction, and 100 Handwritten Mathematical Expression
    Recognition.

    `answer` is a LIST of acceptable strings, all of which are kept: the
    official metric counts a prediction correct when ANY of them appears in it
    (case-insensitive substring), which is why OCRBench tolerates a model that
    answers in a sentence. That lives in judge_qwenlm.py's
    SUBSTRING_MATCH_BENCHMARKS and contacts no judge model.
    """
    from datasets import load_dataset

    ds = load_dataset("echo840/OCRBench", split="test")
    img_dir = (out_dir / "OCRBench_images").resolve()
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, row in enumerate(tqdm(ds, desc="ocrbench")):
        img = row.get("image")
        q = (row.get("question") or "").strip()
        answers = [str(a).strip() for a in (row.get("answer") or []) if str(a).strip()]
        if img is None or not q or not answers:
            continue
        img_path = img_dir / f"ocrbench_{idx:05d}.jpg"
        _save_image(img, img_path)
        records.append({
            "images": [str(img_path)],
            "query": q,
            "response": answers[0],
            "answers": answers,
            "question_id": idx,
            "image_id": idx,
            "question_type": row.get("question_type") or "",
            "source_dataset": row.get("dataset") or "",
        })
    return records


def prepare_hallusionbench(out_dir):
    """HallusionBench (lmms-lab-encoder/HallusionBench), image split: 951 items.

    Yes/no visual questions probing language-prior hallucination. The companion
    `non_image` split is deliberately excluded -- it has no image, and this
    harness sends an image with every request.

    `gt_answer` is "1"/"0" for yes/no. Scored by the yes/no path (no judge
    model). set_id / figure_id / question_id are carried through so the
    published qAcc (all questions in a pair correct) and fAcc (all questions on
    a figure correct) can be computed later; the plain per-question number is
    aAcc.
    """
    from datasets import load_dataset

    ds = load_dataset("lmms-lab-encoder/HallusionBench", split="image")
    img_dir = (out_dir / "HallusionBench_images").resolve()
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, row in enumerate(tqdm(ds, desc="hallusionbench")):
        img = row.get("image")
        q = (row.get("question") or "").strip()
        gt = str(row.get("gt_answer") or "").strip()
        if img is None or not q or gt not in ("0", "1"):
            continue
        img_path = img_dir / f"hallusion_{idx:05d}.jpg"
        _save_image(img, img_path)
        records.append({
            "images": [str(img_path)],
            "query": q + "\nAnswer the question with Yes or No.",
            "response": "yes" if gt == "1" else "no",
            "question_id": idx,
            "image_id": idx,
            "hb_category": row.get("category") or "",
            "hb_subcategory": row.get("subcategory") or "",
            "hb_set_id": str(row.get("set_id") or ""),
            "hb_figure_id": str(row.get("figure_id") or ""),
            "hb_question_id": str(row.get("question_id") or ""),
        })
    return records


def prepare_amber(out_dir):
    """AMBER discriminative (MM-Hallu/amber-benchmark): 14,216 yes/no items.

    Three subsets: existence (4,924), attribute (7,628), relation (1,664).
    The GENERATIVE split (1,004 captioning items) is excluded on purpose -- its
    ground truth is an object list scored with AMBER's CHAIR/coverage pipeline,
    which is a different apparatus from anything this harness does, and faking
    it with string matching would produce a number that looks like CHAIR and
    is not.

    WARNING on the existence subset: every one of its 4,924 items has
    truth="no". Accuracy alone is therefore gameable -- a model that always
    answers "no" scores 100% on that third of the benchmark. Report F1 over the
    combined discriminative set, which is what the AMBER paper leads with, and
    read the per-subset accuracies with that in mind.
    """
    from datasets import load_dataset

    img_dir = (out_dir / "AMBER_images").resolve()
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    idx = 0
    for subset in ("existence", "attribute", "relation"):
        ds = load_dataset(
            "MM-Hallu/amber-benchmark",
            data_files=f"discriminative-{subset}-00000-of-00001.parquet",
            split="train",
        )
        for row in tqdm(ds, desc=f"amber:{subset}"):
            img = row.get("image")
            q = (row.get("query") or row.get("question") or "").strip()
            truth = str(row.get("truth") or "").strip().lower()
            if img is None or not q or truth not in ("yes", "no"):
                continue
            img_path = img_dir / f"amber_{idx:06d}.jpg"
            _save_image(img, img_path)
            records.append({
                "images": [str(img_path)],
                "query": q + "\nAnswer the question with Yes or No.",
                "response": truth,
                "question_id": idx,
                "image_id": str(row.get("id") or ""),
                "amber_subset": subset,
            })
            idx += 1
    return records


def prepare_chartqa(out_dir):
    """ChartQA (Masry et al. 2022), test split: 2,500 QA pairs over real charts,
    half human-written (human_or_machine=0) and half machine-generated (=1).

    Short free-form answers (numbers, labels, yes/no). Graded with the
    benchmark's own published metric, "relaxed accuracy": numeric answers count
    as correct within 5% relative error, everything else is case-insensitive
    exact match. That lives in judge_qwenlm.py's RELAXED_ACC_BENCHMARKS and uses
    no LLM judge at all, so scores are comparable to published leaderboards.

    Relaxed accuracy compares against a SHORT answer, so the official
    single-word/phrase post-prompt is part of the metric, not a stylistic
    choice -- without it a verbose response never parses as a float and falls
    through to string comparison, scoring ~0.

    The human_or_machine flag is kept so the two halves can be split later --
    the human half is the harder and more reported one.
    """
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceM4/ChartQA", split="test")
    suffix = "\nAnswer the question using a single word or phrase."
    img_dir = out_dir / "ChartQA_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, row in enumerate(tqdm(ds, desc="chartqa")):
        q = (row.get("query") or "").strip()
        label = row.get("label")
        ans = label[0] if isinstance(label, list) and label else label
        if not q or ans is None or str(ans).strip() == "":
            continue
        img = row["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_path = img_dir / f"chartqa_{idx:05d}.jpg"
        if not img_path.exists():
            img.save(img_path, "JPEG", quality=95)
        records.append({
            "images": [str(img_path)],
            "query": q + suffix,
            "response": str(ans).strip(),
            "question_id": idx,
            "image_id": idx,
            "human_or_machine": row.get("human_or_machine"),
        })
    return records



BUILDERS = {
    "cvbench": prepare_cvbench,
    "mme-realworld": prepare_mme_realworld,
    "blink": prepare_blink_full,
    "gqa": prepare_gqa,
    "countqa": prepare_countqa,
    "chartqa": prepare_chartqa,
    "docvqa": prepare_docvqa,
    "ocrbench": prepare_ocrbench,
    "evochart": prepare_evochart,
    "hallusionbench": prepare_hallusionbench,
    "amber": prepare_amber,
    "vstar": prepare_vstar,
    "hrbench-4k": lambda out_dir: prepare_hrbench(out_dir, "hrbench-4k"),
    "hrbench-8k": lambda out_dir: prepare_hrbench(out_dir, "hrbench-8k"),
    "zoombench": prepare_zoombench,
}


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark data from HuggingFace")
    parser.add_argument("--benchmark", required=True, type=str)
    parser.add_argument("--data_dir", default=None, type=str, help="Output directory (default: script dir)")
    args = parser.parse_args()

    benchmark = args.benchmark
    benchmark_json = resolve_benchmark_json(benchmark)
    out_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent
    out_json = out_dir / benchmark_json

    if out_json.exists():
        print(f"Already exists: {out_json}, skipping.")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print(f"Preparing {benchmark} -> {out_json} ...")
    data = BUILDERS[benchmark](out_dir)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Generated: {out_json} (records={len(data)})")


if __name__ == "__main__":
    main()
