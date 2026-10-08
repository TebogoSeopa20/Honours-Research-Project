#!/usr/bin/env python
"""Check COSMOS labeled val/test items for overlap with training data, then re-score
saved runs on the clean subset.

Reference sets: Stage 1 training data (cosmos_train.jsonl), Stage 1 validation data
(cosmos_val.jsonl, used for checkpoint selection), Stage 2 training data
(cosmos_labeled_train.jsonl).

Overlap types: identical image file (MD5), near-identical image (difference hash,
Hamming distance <= 4), identical caption after normalisation, near-identical caption
(word Jaccard >= 0.8)."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from calibrate_threshold import find_best_threshold
from misinfodet.utils import read_jsonl

REFS = {
    "stage1_train": "cosmos_train.jsonl",
    "stage1_val": "cosmos_val.jsonl",
    "stage2_train": "cosmos_labeled_train.jsonl",
}
RUNS = ["base_llava_eval", "diag_stage1_only", "stage2_eval_lmonly"]


def captions(rec):
    return [v for k, v in rec.items() if k.startswith("caption") and isinstance(v, str)]


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def image_fingerprints(paths, label):
    md5s, dh, ok = [], [], []
    for i, p in enumerate(paths):
        try:
            data = Path(p).read_bytes()
            md5s.append(hashlib.md5(data).hexdigest())
            img = Image.open(p).convert("L").resize((9, 8), Image.LANCZOS)
            px = np.asarray(img, dtype=np.int16)
            bits = (px[:, 1:] > px[:, :-1]).flatten()
            dh.append(np.packbits(bits).view(np.uint64)[0])
            ok.append(True)
        except Exception:
            md5s.append(None)
            dh.append(np.uint64(0))
            ok.append(False)
        if (i + 1) % 2000 == 0:
            print(f"  {label}: {i + 1}/{len(paths)} images hashed")
    print(f"  {label}: {len(paths) - sum(ok)} unreadable images")
    return md5s, np.array(dh, dtype=np.uint64), np.array(ok)


def hamming_min(q, refs):
    x = np.bitwise_xor(q[:, None], refs[None, :]).view(np.uint8).reshape(len(q), len(refs), 8)
    return np.unpackbits(x, axis=2).sum(axis=2).min(axis=1)


def caption_overlap(eval_caps, ref_caps):
    ref_norm = [norm(c) for c in ref_caps]
    ref_exact = set(ref_norm)
    ref_sets = [(set(c.split()), len(c.split())) for c in ref_norm if len(c.split()) >= 5]
    exact, near = [], []
    for caps in eval_caps:
        e = n = False
        for c in caps:
            nc = norm(c)
            if nc in ref_exact:
                e = True
            words = set(nc.split())
            if len(words) < 5:
                continue
            for rs, rl in ref_sets:
                if 0.6 * len(words) <= rl <= 1.67 * len(words) and len(words & rs) / len(words | rs) >= 0.8:
                    n = True
                    break
        exact.append(e)
        near.append(n or e)
    return np.array(exact), np.array(near)


def auroc(scored):
    from sklearn.metrics import roc_auc_score

    return roc_auc_score([g for _, g in scored], [s for s, _ in scored])


def acc(scored, t):
    return sum(1 for s, g in scored if (s >= t) == g) / len(scored)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--pred-dir", default="outputs/predictions")
    ap.add_argument("--max-hamming", type=int, default=4)
    args = ap.parse_args()
    d = Path(args.data_dir)

    evals = {s: read_jsonl(d / f"cosmos_labeled_{s}.jsonl") for s in ("val", "test")}
    eval_md5, eval_dh, eval_ok = {}, {}, {}
    for s, recs in evals.items():
        eval_md5[s], eval_dh[s], eval_ok[s] = image_fingerprints([r["image_path"] for r in recs], f"eval {s}")

    flags = {s: np.zeros(len(r), dtype=bool) for s, r in evals.items()}
    print("\n=== Overlap of labeled val/test items with each reference set ===")
    for name, fname in REFS.items():
        refs = read_jsonl(d / fname)
        r_md5, r_dh, r_ok = image_fingerprints([r["image_path"] for r in refs], name)
        r_dh = r_dh[r_ok]
        r_md5_set = {m for m in r_md5 if m}
        r_caps = [c for r in refs for c in captions(r)]
        for s, recs in evals.items():
            img_exact = np.array([m in r_md5_set for m in eval_md5[s]])
            img_near = (hamming_min(eval_dh[s], r_dh) <= args.max_hamming) & eval_ok[s]
            cap_exact, cap_near = caption_overlap([captions(r) for r in recs], r_caps)
            anyf = img_exact | img_near | cap_exact | cap_near
            labels = np.array([r["label"] for r in recs])
            print(f"{s:4s} vs {name:12s} (n_ref={len(refs):5d}): "
                  f"img exact {img_exact.sum():3d} | img near {img_near.sum():3d} | "
                  f"cap exact {cap_exact.sum():3d} | cap near {cap_near.sum():3d} | "
                  f"ANY {anyf.sum():3d}/{len(recs)} (real {anyf[labels == 0].sum()}, ooc {anyf[labels == 1].sum()})")
            flags[s] |= anyf

    out = {s: [f"{s}_{i}" for i in np.where(f)[0]] for s, f in flags.items()}
    Path(args.pred_dir).mkdir(parents=True, exist_ok=True)
    Path(args.pred_dir, "leakage_flags.json").write_text(json.dumps(out, indent=1))
    print(f"\nFlagged (any overlap, any reference set): val {flags['val'].sum()}/{len(flags['val'])}, "
          f"test {flags['test'].sum()}/{len(flags['test'])}  -> {args.pred_dir}/leakage_flags.json")

    print("\n=== Re-scoring saved runs: all test items vs clean test items ===")
    clean = {s: set(f"{s}_{i}" for i in np.where(~f)[0]) for s, f in flags.items()}
    for run in RUNS:
        p = Path(args.pred_dir) / f"{run}_cosmos_scores.jsonl"
        if not p.exists():
            print(f"{run}: no score file, skipped")
            continue
        rows = read_jsonl(p)
        val_all = [(r["score"], r["label"]) for r in rows if r["split"] == "val"]
        test_all = [(r["score"], r["label"]) for r in rows if r["split"] == "test"]
        val_c = [(r["score"], r["label"]) for r in rows if r["split"] == "val" and r["id"] in clean["val"]]
        test_c = [(r["score"], r["label"]) for r in rows if r["split"] == "test" and r["id"] in clean["test"]]
        t_all, _ = find_best_threshold(val_all)
        t_c, _ = find_best_threshold(val_c)
        print(f"{run:20s} ALL   n={len(test_all):3d} AUROC {auroc(test_all):.4f} acc {acc(test_all, t_all):.4f} (t={t_all:.2f})")
        print(f"{'':20s} CLEAN n={len(test_c):3d} AUROC {auroc(test_c):.4f} acc {acc(test_c, t_c):.4f} (t={t_c:.2f})")


if __name__ == "__main__":
    main()
