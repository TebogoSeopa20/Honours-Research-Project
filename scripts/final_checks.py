#!/usr/bin/env python
"""Final CPU-only checks.

  coverage      evidence coverage and length in the cached retrieval run
  sbert         text-only baseline: caption-pair dissimilarity (sentence embeddings)
  explanations  self-contradiction and image-reference counts; blinded rating sheets
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path

PRED = Path("outputs/predictions")
DATA = Path("data")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def coverage(args):
    rows = read_jsonl(PRED / f"{args.run}_retrieval_cosmos_scores.jsonl")
    for split in ("val", "test"):
        r = [x for x in rows if x["split"] == split]
        n_ev = [len(x["evidence"]) for x in r]
        words = [sum(len(f"{e.get('title', '')} {e.get('snippet', '')}".split()) for e in x["evidence"]) for x in r]
        nonempty = [w for w in words if w > 0]
        print(f"{split}: items {len(r)} | with evidence {sum(1 for n in n_ev if n > 0)} | "
              f"snippets/item {sum(n_ev) / len(r):.2f} | evidence words/item: mean {sum(words) / len(r):.1f}, "
              f"min {min(words)}, max {max(words)} | items under 20 words {sum(1 for w in words if w < 20)}")
    print("\nExample (test_0):")
    ex = next(x for x in rows if x["id"] == "test_0")
    print("queries:", ex["queries"])
    for e in ex["evidence"]:
        print("-", (e.get("title", "") + " | " + e.get("snippet", ""))[:200])


def sbert(args):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import roc_auc_score

    model = SentenceTransformer(args.model)
    out = PRED / "sbert_caption_sim_cosmos_scores.jsonl"
    rows, scored = [], {}
    for split in ("val", "test"):
        recs = read_jsonl(DATA / f"cosmos_labeled_{split}.jsonl")
        e1 = model.encode([r["caption1"] for r in recs], normalize_embeddings=True)
        e2 = model.encode([r["caption2"] for r in recs], normalize_embeddings=True)
        sims = (e1 * e2).sum(axis=1)
        scored[split] = []
        for i, (r, s) in enumerate(zip(recs, sims)):
            score = float(min(max(1 - s, 0.0), 1.0))
            rows.append({"id": f"{split}_{i}", "split": split, "score": score, "label": r["label"]})
            scored[split].append((score, r["label"]))
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

    def macro_f1(t, s):
        p = [int(x >= t) for x, _ in s]
        g = [y for _, y in s]
        f = []
        for c in (0, 1):
            tp = sum(a == c and b == c for a, b in zip(p, g))
            fp = sum(a == c and b != c for a, b in zip(p, g))
            fn = sum(a != c and b == c for a, b in zip(p, g))
            pr, rc = (tp / (tp + fp) if tp + fp else 0), (tp / (tp + fn) if tp + fn else 0)
            f.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
        return sum(f) / 2

    cands = sorted({round(x, 3) for x, _ in scored["val"]})
    t = max(cands, key=lambda c: macro_f1(c, scored["val"]))
    test = scored["test"]
    acc = sum((x >= t) == y for x, y in test) / len(test)
    print(f"SBERT ({args.model}) caption dissimilarity")
    print(f"val AUROC {roc_auc_score([y for _, y in scored['val']], [x for x, _ in scored['val']]):.4f}")
    print(f"TEST AUROC {roc_auc_score([y for _, y in test], [x for x, _ in test]):.4f} | "
          f"acc {acc:.4f} | macro F1 {macro_f1(t, test):.4f} (threshold {t:.3f}, from val)")
    print(f"Scores written to {out}")


OOC_WORDS = r"out[- ]of[- ]context|different (events?|people|places|contexts?)|do(es)? not (match|correspond)|inconsistent|mismatch|unrelated|contradict"
CONS_WORDS = r"\bconsistent\b|both captions (can|could|are|accurately|correctly|genuinely) |not out[- ]of[- ]context|same event|align(s|ed)? with"
IMAGE_WORDS = r"\b(in the image|the image shows|image depicts|visible|pictured|can be seen|we can see|in the photo|the photo shows|background|foreground|wearing|standing)\b"


def explanations(args):
    rows = read_jsonl(Path("outputs/explanations/explanations_cosmos_test.jsonl"))
    contra, img = [], 0
    for r in rows:
        t = r["explanation"].lower()
        has_ooc, has_cons = bool(re.search(OOC_WORDS, t)), bool(re.search(CONS_WORDS, t))
        if (r["verdict"] == 1 and has_cons and not has_ooc) or (r["verdict"] == 0 and has_ooc and not has_cons):
            contra.append(r["id"])
        img += bool(re.search(IMAGE_WORDS, t))
    n = len(rows)
    print(f"explanations: {n}")
    print(f"refer to visual content: {img} ({img / n:.1%})")
    print(f"possible self-contradiction (wording supports the opposite verdict only): {len(contra)} ({len(contra) / n:.1%})")
    for split_name, sel in (("correct verdicts", 1), ("wrong verdicts", 0)):
        sub = [r for r in rows if r["correct"] == sel]
        c = sum(1 for r in sub if r["id"] in contra)
        print(f"  {split_name}: {len(sub)} items, {c} flagged")
    print("flagged ids (check by hand):", contra[:30])

    out = Path("outputs/explanations")
    sheet = list(csv.DictReader(open(out / "rating_sheet.csv", encoding="utf-8")))
    cols = ["item", "image_file", "caption1", "caption2", "verdict", "explanation",
            "factual_accuracy_1to5", "coherence_1to5", "relevance_1to5", "notes"]
    for name, items in (("rating_sheet_blind.csv", sheet), ("rating_sheet_rater2.csv", sheet[: args.n_rater2])):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in items:
                w.writerow({k: r.get(k, "") for k in cols})
    with open(out / "rating_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "gold", "correct"])
        for r in sheet:
            w.writerow([r["item"], r["gold"], r["correct"]])
    print(f"\nwrote rating_sheet_blind.csv ({len(sheet)} items), rating_sheet_rater2.csv ({min(args.n_rater2, len(sheet))} items), "
          f"rating_key.csv (do not open until rating is finished)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coverage")
    c.add_argument("--run", default="diag_stage1_only")
    s = sub.add_parser("sbert")
    s.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    e = sub.add_parser("explanations")
    e.add_argument("--n-rater2", type=int, default=20)
    args = ap.parse_args()
    {"coverage": coverage, "sbert": sbert, "explanations": explanations}[args.cmd](args)


if __name__ == "__main__":
    main()
