"""
WhereOPD scene generator: synthetic counting scenes with an object-location hint.

Each scene asks for the count of one (colour, shape) pair among distractors.
Every row stores, besides the question and the image:
  extra_info["where_hint"]  the object-location hint the WhereOPD teacher sees
                            ("Scanning for brown crosses: found one near (929, 298),
                            ... Counting: 3 total.")
  data_source               "whereopd_counting" (registered in verl's reward registry)

Scene recipe: 7 shapes (circle, square, triangle, diamond, star, cross, hexagon),
12 colours, 10 backgrounds with per-scene +-8 RGB jitter, 1024x1024 images,
a CIELAB dE >= 27 object-vs-background contrast filter, and a per-scene
vocabulary of 2-5 colours x 2-4 shapes. JPEG bytes are embedded in the parquet
(row_group_size=256).

The exact settings of the WhereOPD datasets are recorded in build_dataset.py;
build them with that script rather than calling this one directly.
"""

import argparse
import io
import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

COLORS: Dict[str, Tuple[int, int, int]] = {
    "red":    (220,  50,  47),
    "green":  (133, 153,   0),
    "blue":   ( 38, 139, 210),
    "yellow": (181, 137,   0),
    "purple": (108, 113, 196),
    "orange": (203,  75,  22),
    "pink":   (211,  54, 130),
    "brown":  (121,  85,  61),
    "teal":   ( 42, 161, 152),
    "gray":   (131, 148, 150),
    "black":  ( 35,  38,  39),
    "white":  (253, 246, 227),
}

# grounds: light neutrals, dark neutrals, muted tints — nameable-safe (objects
# are never described by background, so tints only need to be distinct grounds)
BACKGROUNDS: List[Tuple[int, int, int]] = [
    (245, 245, 245),   # light grey
    (253, 246, 227),   # cream
    (222, 226, 230),   # cool light grey
    (207, 216, 220),   # blue-grey
    (230, 220, 205),   # sand
    (215, 227, 213),   # pale sage
    (219, 213, 229),   # pale lavender
    ( 60,  63,  65),   # dark slate
    ( 40,  44,  52),   # near-black blue
    ( 84,  72,  64),   # dark umber
]

SHAPES = ("circle", "square", "triangle", "diamond", "star", "cross", "hexagon")
PLURAL = {s: s + ("es" if s == "cross" else "s") for s in SHAPES}


# --------------------------------------------------------------------- color
def _srgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    c = np.where(rgb01 > 0.04045, ((rgb01 + 0.055) / 1.055) ** 2.4, rgb01 / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16,
                     500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], axis=1)


def delta_e(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    lab = _srgb_to_lab(np.array([a, b], dtype=np.float64) / 255.0)
    return float(np.sqrt(((lab[0] - lab[1]) ** 2).sum()))


MIN_BG_DE = 27.0   # object colors closer than this to the ground are excluded


# ------------------------------------------------------------------ geometry
class SpatialGrid:
    def __init__(self, width, height, cell_size):
        self.cell = max(4, cell_size)
        self.gx = (width + self.cell - 1) // self.cell
        self.gy = (height + self.cell - 1) // self.cell
        self.cells = [[[] for _ in range(self.gx)] for _ in range(self.gy)]

    def _idx(self, x, y): return x // self.cell, y // self.cell

    def neighbors(self, x, y):
        cx, cy = self._idx(x, y)
        out = []
        for yy in range(max(0, cy - 1), min(self.gy, cy + 2)):
            for xx in range(max(0, cx - 1), min(self.gx, cx + 2)):
                out.extend(self.cells[yy][xx])
        return out

    def add(self, x, y, r):
        cx, cy = self._idx(x, y)
        self.cells[cy][cx].append((x, y, r))


def overlaps(x, y, r, others):
    return any((x - ox) ** 2 + (y - oy) ** 2 < (r + orr) ** 2 for ox, oy, orr in others)


def _poly(cx, cy, r, n, rot=-math.pi / 2):
    return [(cx + r * math.cos(rot + 2 * math.pi * k / n),
             cy + r * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]


def draw_object(draw, shape, x, y, r, rgb):
    if shape == "circle":
        draw.ellipse((x - r, y - r, x + r, y + r), fill=rgb)
    elif shape == "square":
        s = r / math.sqrt(2)                 # inscribe: keep footprint ~= r
        draw.rectangle((x - s, y - s, x + s, y + s), fill=rgb)
    elif shape == "triangle":
        draw.polygon(_poly(x, y, r, 3), fill=rgb)
    elif shape == "diamond":
        draw.polygon([(x, y - r), (x + r * 0.7, y), (x, y + r), (x - r * 0.7, y)], fill=rgb)
    elif shape == "star":
        pts = []
        for k in range(10):
            rr = r if k % 2 == 0 else r * 0.45
            a = -math.pi / 2 + math.pi * k / 5
            pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
        draw.polygon(pts, fill=rgb)
    elif shape == "cross":
        w = r * 0.38
        draw.rectangle((x - w, y - r, x + w, y + r), fill=rgb)
        draw.rectangle((x - r, y - w, x + r, y + w), fill=rgb)
    elif shape == "hexagon":
        draw.polygon(_poly(x, y, r, 6, rot=0), fill=rgb)
    else:
        raise ValueError(shape)


# --------------------------------------------------------------------- scene
Object = Tuple[str, str, int, int, int]




def pick_background(rng) -> Tuple[int, int, int]:
    base = rng.choice(BACKGROUNDS)
    return tuple(max(0, min(255, c + rng.randint(-8, 8))) for c in base)


def generate_scene(rng, width, height, min_objects, max_objects,
                   min_radius, max_radius, margin, max_tries_per_obj):
    bg = pick_background(rng)
    usable = [c for c in COLORS if delta_e(COLORS[c], bg) >= MIN_BG_DE]
    # PER-SCENE VOCABULARY: with all 12 colors x 7 shapes active, 10-80 objects
    # spread over 84 pairs leaves nearly every pair at count 0-2 and the
    # question degenerates to "how many X? ... 1". Each scene uses a small
    # random subset (2-5 colors x 2-4 shapes -> 4-20 pairs), so per-pair counts
    # land back in the 3-15 range while the DATASET still spans all 84 pairs
    # and all backgrounds.
    scene_colors = rng.sample(usable, min(len(usable), rng.randint(2, 5)))
    scene_shapes = rng.sample(SHAPES, rng.randint(2, 4))
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    target_n = rng.randint(min_objects, max_objects)
    grid = SpatialGrid(width, height, cell_size=2 * max_radius + 2)
    objects: List[Object] = []
    attempts = 0
    while len(objects) < target_n and attempts < target_n * max_tries_per_obj:
        attempts += 1
        shape = rng.choice(scene_shapes)
        color = rng.choice(scene_colors)
        r = rng.randint(min_radius, max_radius)
        x = rng.randint(margin + r, width - margin - r)
        y = rng.randint(margin + r, height - margin - r)
        if overlaps(x, y, r, grid.neighbors(x, y)):
            continue
        draw_object(draw, shape, x, y, r, COLORS[color])
        grid.add(x, y, r)
        objects.append((color, shape, x, y, r))
    return img, objects, bg


# ------------------------------------------------------------------- records
LETTERS = ["A", "B", "C", "D"]


def make_distractors(correct, rng):
    offsets = [-3, -2, -1, 1, 2, 3]
    rng.shuffle(offsets)
    out = []
    for off in offsets:
        cand = correct + off
        if cand > 0 and cand not in out and cand != correct:
            out.append(cand)
        if len(out) == 3:
            return out
    extra = correct + 4
    while len(out) < 3:
        if extra not in out and extra != correct:
            out.append(extra)
        extra += 1
    return out


def make_mcq(correct, rng):
    vals = [correct] + make_distractors(correct, rng)
    rng.shuffle(vals)
    options = {l: str(v) for l, v in zip(LETTERS, vals)}
    return options, LETTERS[vals.index(correct)]


def make_prompt(color, shape, options):
    opts = "\n".join(f"{l}. {v}" for l, v in options.items())
    return (f"<image>\nHow many {color} {PLURAL[shape]} are in the image?\n\n"
            f"{opts}\n\nAnswer with only the letter (A, B, C, or D), nothing else.")


def make_where_hint(color, shape, coords):
    """The teacher's privilege: a search procedure around the target coordinates,
    ending in the count, e.g. "Scanning for brown crosses: found one near
    (929, 298), found one near (312, 760). Counting: 2 total."
    """
    if not coords:
        return f"Scanning for {color} {PLURAL[shape]}: none found. Counting: 0 total."
    found = ", ".join(f"found one near ({x}, {y})" for x, y, _ in coords)
    return f"Scanning for {color} {PLURAL[shape]}: {found}. Counting: {len(coords)} total."


def jpeg_bytes(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="whereopd_scenes")
    ap.add_argument("--n_images", type=int, default=11000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--min_objects", type=int, default=25)
    ap.add_argument("--max_objects", type=int, default=150)
    ap.add_argument("--min_radius", type=int, default=7)
    ap.add_argument("--max_radius", type=int, default=20)
    ap.add_argument("--margin", type=int, default=10)
    ap.add_argument("--max_tries_per_obj", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    records = []
    from collections import Counter
    bg_hist, shape_hist = Counter(), Counter()

    for i in tqdm(range(args.n_images), desc="WhereOPD scenes"):
        img, objects, bg = generate_scene(
            rng, args.width, args.height, args.min_objects, args.max_objects,
            args.min_radius, args.max_radius, args.margin, args.max_tries_per_obj)
        if not objects:
            continue
        counts: Dict[str, int] = {}
        coords: Dict[str, list] = {}
        for color, shape, x, y, r in objects:
            k = f"{color}_{shape}"
            counts[k] = counts.get(k, 0) + 1
            coords.setdefault(k, []).append((x, y, r))
            shape_hist[shape] += 1
        # target: a present pair, chosen with probability proportional to its
        # count -- the campaign's weakest axis is HIGH counts, so the question
        # distribution leans toward the dense pairs rather than singletons.
        keys=list(counts.keys())
        key = rng.choices(keys, weights=[counts[k] for k in keys], k=1)[0]
        t_color, t_shape = key.split("_")
        options, letter = make_mcq(counts[key], rng)
        prompt = make_prompt(t_color, t_shape, options)
        answer, qtype = letter, "mcq"
        hint = make_where_hint(t_color, t_shape, coords[key])
        bg_hist[str(bg)] += 0  # bg jittered per scene; bucket by base below
        records.append({
            "data_source": "whereopd_counting",
            "prompt": [{"role": "user", "content": prompt}],
            "images": [{"bytes": jpeg_bytes(img)}],
            "ability": "visual_question_answering",
            "reward_model": {"style": "none", "ground_truth": answer},
            "extra_info": {
                "answer": answer, "question": prompt, "question_type": qtype,
                "where_hint": hint,
                "source_extra_info": {
                    "image_index": i, "target_color": t_color,
                    "target_shape": t_shape, "target_count": counts[key],
                    "options": options, "total_objects": len(objects),
                    "all_counts": counts, "background": list(bg),
                    "image_size": {"width": args.width, "height": args.height},
                    "target_coords": [[x, y] for x, y, _ in coords[key]],
                },
            },
        })

    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(records),
                   os.path.join(args.out_dir, "train.parquet"),
                   row_group_size=256)
    size = os.path.getsize(os.path.join(args.out_dir, "train.parquet")) / 1e6
    print(f"Wrote {args.out_dir}/train.parquet: {len(records)} records, {size:.0f} MB")
    print("shape usage:", dict(shape_hist))


if __name__ == "__main__":
    main()
