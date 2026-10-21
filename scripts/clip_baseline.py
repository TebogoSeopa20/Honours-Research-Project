#!/usr/bin/env python
"""CLIP image-caption similarity baseline (NewsCLIPpings, Luo et al. 2021).

  newsclippings  full merged_balanced test set from the authors' released CLIP ViT-B/32 embeddings (CPU)
  cosmos         COSMOS labelled val/test, CLIP ViT-B/32 computed on the images (GPU recommended)
  mmfakebench    same stratified sample as calibrate_mmfakebench.py (seed 42, n 2000)

Thresholds and logistic combinations are always fitted on val / calibration data and applied unchanged to test."""
import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PRED = Path("outputs/predictions")
DATA = Path("data")
CLIP_ID = "openai/clip-vit-base-patch32"


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def macro_f1(y, pred):
    f = []
    for c in (0, 1):
        tp = sum(a == c and b == c for a, b in zip(pred, y))
        fp = sum(a == c and b != c for a, b in zip(pred, y))
        fn = sum(a != c and b == c for a, b in zip(pred, y))
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f) / 2


def evaluate(name, yv, sv, yt, st):
    yv, sv, yt, st = map(np.asarray, (yv, sv, yt, st))
    cands = np.unique(np.quantile(sv, np.linspace(0, 1, 401)))
    t = max(cands, key=lambda c: macro_f1(yv, (sv >= c).astype(int)))
    pt = (st >= t).astype(int)
    print(f"{name:42s} val AUROC {roc_auc_score(yv, sv):.4f} | TEST AUROC {roc_auc_score(yt, st):.4f} "
          f"| acc {np.mean(pt == yt):.4f} | macro F1 {macro_f1(yt, pt):.4f} (threshold {t:.3f} from val)")


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def load_emb(path):
    obj = pickle.load(open(path, "rb"))
    if isinstance(obj, dict):
        return {str(k): np.asarray(v).reshape(-1) for k, v in obj.items()}
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            idcol = next((c for c in obj.columns if str(c).lower() in ("id", "image_id")), None)
            ids = obj[idcol] if idcol is not None else obj.index
            vals = obj.drop(columns=[idcol]) if idcol is not None else obj
            if vals.shape[1] == 1:
                arr = np.stack(vals.iloc[:, 0].map(np.asarray).values)
            else:
                arr = vals.values
            return {str(i): v for i, v in zip(ids, arr)}
    except ImportError:
        pass
    raise SystemExit(f"unrecognised embedding format in {path}: {type(obj)}")


def newsclippings(args):
    root = Path(args.ncp_dir)
    feats, cos, ys = {}, {}, {}
    for split in ("val", "test"):
        anns = json.load(open(root / "data" / "merged_balanced" / f"{split}.json"))["annotations"]
        img = load_emb(root / "embeddings" / "clip_image_embeddings" / f"clip_image_embeddings_{split}.pkl")
        txt = load_emb(root / "embeddings" / "clip_text_embeddings" / f"clip_text_embeddings_{split}.pkl")
        keep = [a for a in anns if str(a["image_id"]) in img and str(a["id"]) in txt]
        print(f"{split}: {len(keep)}/{len(anns)} pairs with both embeddings")
        I = norm([img[str(a["image_id"])] for a in keep])
        T = norm([txt[str(a["id"])] for a in keep])
        cos[split] = (I * T).sum(1)
        feats[split] = np.hstack([I * T, np.abs(I - T)])
        ys[split] = np.array([int(bool(a["falsified"])) for a in keep])
    print("\nNewsCLIPpings merged_balanced (label 1 = falsified)")
    evaluate("CLIP zero-shot (1 - cosine)", ys["val"], 1 - cos["val"], ys["test"], 1 - cos["test"])
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(feats["val"], ys["val"])
    s_val = clf.predict_proba(feats["val"])[:, 1]
    s_test = clf.predict_proba(feats["test"])[:, 1]
    print("(logistic probe fitted on val, so its val AUROC is optimistic)")
    evaluate("CLIP features + logistic probe (fit on val)", ys["val"], s_val, ys["test"], s_test)


def clip_model():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return CLIPModel.from_pretrained(CLIP_ID).to(dev).eval(), CLIPProcessor.from_pretrained(CLIP_ID), dev


def embed_images(model, proc, dev, paths, bs=32):
    import torch
    from PIL import Image

    out = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        with torch.no_grad():
            e = model.get_image_features(**proc(images=ims, return_tensors="pt").to(dev))
        out.append(e.float().cpu().numpy())
        print(f"  images {min(i + bs, len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    return norm(np.vstack(out))


def embed_texts(model, proc, dev, texts, bs=64):
    import torch

    out = []
    for i in range(0, len(texts), bs):
        inp = proc(text=texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)
        with torch.no_grad():
            out.append(model.get_text_features(**inp).float().cpu().numpy())
    return norm(np.vstack(out))


def cosmos(args):
    model, proc, dev = clip_model()
    S, Y = {}, {}
    for split in ("val", "test"):
        recs = read_jsonl(DATA / f"cosmos_labeled_{split}.jsonl")
        I = embed_images(model, proc, dev, [r["image_path"] for r in recs])
        C1 = embed_texts(model, proc, dev, [r["caption1"] for r in recs])
        C2 = embed_texts(model, proc, dev, [r["caption2"] for r in recs])
        s1, s2, cc = (I * C1).sum(1), (I * C2).sum(1), (C1 * C2).sum(1)
        S[split] = {"gap": np.abs(s1 - s2), "text": 1 - cc, "min": -np.minimum(s1, s2),
                    "X": np.stack([s1, s2, np.abs(s1 - s2), cc, np.minimum(s1, s2)], 1)}
        Y[split] = np.array([r["label"] for r in recs])
    print("\nCOSMOS labelled test (label 1 = out-of-context)")
    evaluate("image-caption gap |cos(I,C1) - cos(I,C2)|", Y["val"], S["val"]["gap"], Y["test"], S["test"]["gap"])
    evaluate("weaker caption match -min(cos(I,C))", Y["val"], S["val"]["min"], Y["test"], S["test"]["min"])
    evaluate("caption-caption dissimilarity (CLIP text)", Y["val"], S["val"]["text"], Y["test"], S["test"]["text"])
    clf = LogisticRegression(max_iter=2000).fit(S["val"]["X"], Y["val"])
    sv, st = clf.predict_proba(S["val"]["X"])[:, 1], clf.predict_proba(S["test"]["X"])[:, 1]
    evaluate("CLIP combined (logistic, fit on val)", Y["val"], sv, Y["test"], st)
    out = PRED / "clip_cosmos_scores.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for split, s in (("val", sv), ("test", st)):
            for i, (x, y) in enumerate(zip(s, Y[split])):
                f.write(json.dumps({"id": f"{split}_{i}", "split": split, "score": float(x), "label": int(y)}) + "\n")
    print(f"Scores written to {out} (same ids as the LLaVA runs, usable with compare_runs.py)")


def stratified_sample(records, n, seed):
    by_type = defaultdict(list)
    for r in records:
        by_type[r["forgery_type"]].append(r)
    rng = random.Random(seed)
    calib, evals = [], []
    for ftype, items in sorted(by_type.items()):
        k = round(n * len(items) / len(records))
        picked = rng.sample(items, k)
        calib += picked[:k // 2]
        evals += picked[k // 2:]
    rng.shuffle(calib)
    rng.shuffle(evals)
    return calib, evals


def mmfakebench(args):
    model, proc, dev = clip_model()
    calib, evals = stratified_sample(read_jsonl(DATA / "mmfakebench_test.jsonl"), args.n, args.seed)
    sc = {}
    for name, recs in (("calib", calib), ("eval", evals)):
        I = embed_images(model, proc, dev, [r["image_path"] for r in recs])
        T = embed_texts(model, proc, dev, [r["caption"] for r in recs])
        sc[name] = 1 - (I * T).sum(1)
    yc = [r["label"] for r in calib]
    ye = [r["label"] for r in evals]
    print("\nMMFakeBench sample (label 1 = any fake)")
    evaluate("CLIP zero-shot (1 - cosine)", yc, sc["calib"], ye, sc["eval"])
    real = [i for i, r in enumerate(evals) if r["forgery_type"] == "original"]
    for ft in sorted({r["forgery_type"] for r in evals} - {"original"}):
        idx = [i for i, r in enumerate(evals) if r["forgery_type"] == ft]
        y = [1] * len(idx) + [0] * len(real)
        print(f"  {ft:32s} AUROC vs original: {roc_auc_score(y, [sc['eval'][i] for i in idx + real]):.3f}")
    out = PRED / "clip_mmfakebench_scores.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for recs, s in ((calib, sc["calib"]), (evals, sc["eval"])):
            for r, x in zip(recs, s):
                f.write(json.dumps({"id": r["id"], "score": float(x)}) + "\n")
    print(f"Scores written to {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("newsclippings")
    p.add_argument("--ncp-dir", required=True, help="folder containing data/ and embeddings/")
    sub.add_parser("cosmos")
    p = sub.add_parser("mmfakebench")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    {"newsclippings": newsclippings, "cosmos": cosmos, "mmfakebench": mmfakebench}[args.cmd](args)


if __name__ == "__main__":
    main()
