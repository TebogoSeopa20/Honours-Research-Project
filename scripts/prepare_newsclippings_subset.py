#!/usr/bin/env python
"""Build a stratified NewsCLIPpings (merged_balanced) subset by streaming VisualNews origin.tar
and keeping only the images needed. Writes data/newsclippings_{val,test}.jsonl with
{"id", "image_path", "caption", "label"} (label 1 = falsified / out-of-context).

Passes: --probe lists the first archive members; pass 1 streams until data.json is found;
pass 2 streams and extracts the sampled images, stopping once all are found."""
import argparse
import json
import random
import tarfile
import time
import urllib.request
from pathlib import Path


def stream(url):
    return tarfile.open(fileobj=urllib.request.urlopen(url), mode="r|")


def norm(p):
    p = p.lstrip("./")
    return p[len("origin/"):] if p.startswith("origin/") else p


def probe(url, k):
    for i, m in enumerate(stream(url)):
        print(m.name, m.size)
        if i + 1 >= k:
            break


def sample(anns, n, seed):
    rng = random.Random(seed)
    pos = [a for a in anns if a["falsified"]]
    neg = [a for a in anns if not a["falsified"]]
    k = n // 2
    return rng.sample(pos, min(k, len(pos))) + rng.sample(neg, min(n - k, len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://www.cs.rice.edu/~vo9/visualnews/origin.tar")
    ap.add_argument("--annotations", default="news_clippings/data/merged_balanced")
    ap.add_argument("--out", default="data/newsclippings")
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe", type=int, default=0)
    args = ap.parse_args()

    if args.probe:
        probe(args.url, args.probe)
        return

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    dj = out / "visualnews_data.json"
    if not dj.exists():
        tf = stream(args.url)
        t0 = time.time()
        for i, m in enumerate(tf):
            if m.isfile() and m.name.endswith("data.json"):
                dj.write_bytes(tf.extractfile(m).read())
                print(f"found {m.name} after {i + 1} members, {time.time() - t0:.0f}s")
                break
            if (i + 1) % 200000 == 0:
                print(f"  scanning for data.json: {i + 1} members, {time.time() - t0:.0f}s", flush=True)
        else:
            raise SystemExit("data.json not found in archive")

    vn = {str(e["id"]): e for e in json.loads(dj.read_text())}
    splits = {}
    for split, n in (("test", args.n_test), ("val", args.n_val)):
        anns = json.loads((Path(args.annotations) / f"{split}.json").read_text())["annotations"]
        splits[split] = sample(anns, n, args.seed)

    need = {}
    for picked in splits.values():
        for a in picked:
            need[norm(vn[str(a["image_id"])]["image_path"])] = None
    todo = {p for p in need if not (out / "images" / p).exists()}
    print(f"{len(need)} images needed, {len(todo)} still to extract", flush=True)

    if todo:
        tf = stream(args.url)
        t0 = time.time()
        for i, m in enumerate(tf):
            key = norm(m.name)
            if m.isfile() and key in todo:
                dest = out / "images" / key
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tf.extractfile(m).read())
                todo.discard(key)
                if not todo:
                    print(f"all images extracted after {i + 1} members, {time.time() - t0:.0f}s")
                    break
            if (i + 1) % 200000 == 0:
                print(f"  {i + 1} members scanned, {len(todo)} images left, {time.time() - t0:.0f}s", flush=True)
        if todo:
            print(f"WARNING: {len(todo)} images not found in archive; those items are dropped")

    for split, picked in splits.items():
        rows = []
        for a in picked:
            key = norm(vn[str(a["image_id"])]["image_path"])
            img = out / "images" / key
            if not img.exists():
                continue
            rows.append({"id": f"ncp_{split}_{a['id']}_{a['image_id']}", "image_path": str(img),
                         "caption": vn[str(a["id"])]["caption"], "label": int(bool(a["falsified"]))})
        path = Path("data") / f"newsclippings_{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{split}: wrote {len(rows)} records to {path}  (falsified={sum(r['label'] for r in rows)})")


if __name__ == "__main__":
    main()
