"""The WhereOPD evaluation suite: the 15 benchmarks the eval scripts accept.

Order is the paper's table order. Each entry: (name used on the command line,
benchmark JSON built by prepare_data.py, short name used in the tables).
"""
SUITE = [
    ("cvbench",        "cvbench.json",        "CVB"),
    ("mme-realworld",  "MME_RealWorld.json",  "MME-RW"),
    ("blink",          "blink_full.json",     "BLINK"),
    ("gqa",            "gqa.json",            "GQA"),
    ("countqa",        "countqa.json",        "CountQA"),
    ("chartqa",        "chartqa.json",        "ChartQA"),
    ("docvqa",         "docvqa.json",         "DocVQA"),
    ("ocrbench",       "ocrbench.json",       "OCRB"),
    ("evochart",       "evochart.json",       "EvoChart"),
    ("hallusionbench", "hallusionbench.json", "HallB"),
    ("amber",          "amber.json",          "AMBER"),
    ("vstar",          "vstar.json",          "V*"),
    ("hrbench-4k",     "hr_bench_4k.json",    "HR-4K"),
    ("hrbench-8k",     "hr_bench_8k.json",    "HR-8K"),
    ("zoombench",      "zoombench.json",      "Zoom"),
]
SUITE_NAMES = [name for name, _, _ in SUITE]
BENCH_JSON = {name: js for name, js, _ in SUITE}
