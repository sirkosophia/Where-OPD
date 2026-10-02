# WhereOPD data

Synthetic counting scenes (coloured shapes on a 1024x1024 canvas) with a
`where_hint` per scene: the target objects' locations and the count, which only
the teacher sees.

| dataset | used for | built by |
|---|---|---|
| `whereopd_count_mcq_3k` | Qwen3.5-4B training | `generate_scenes.py` |
| `whereopd_count_open_3k` | Qwen3.5-9B training  | `make_open_ended.py` from the mcq set |


```bash
python build_dataset.py build whereopd_count_mcq_3k  --root /path/to/whereopd_data
python build_dataset.py build whereopd_count_open_3k --root /path/to/whereopd_data
```

