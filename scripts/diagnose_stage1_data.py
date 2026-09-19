#!/usr/bin/env python
"""Diagnose Stage 1 training data quality: duplicate records, and how many
times each apparent named entity actually appears. No GPU needed — pure
text analysis on the already-prepared cosmos_train.jsonl.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


def extract_entity_phrases(caption: str) -> list[str]:
    """Same rough heuristic as the comparison script: consecutive
    capitalized words, e.g. 'Julian Castro', 'Mark Zuckerberg'."""
    words = caption.split()
    phrases = []
    i = 0
    while i < len(words) - 1:
        if words[i][:1].isupper() and words[i + 1][:1].isupper():
            phrase = words[i] + " " + words[i + 1]
            phrases.append(re.sub(r"[.,;:]$", "", phrase))
            i += 2
        else:
            i += 1
    return phrases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl_path", default="data/cosmos_train.jsonl", nargs="?")
    args = ap.parse_args()

    total = 0
    seen_images = set()
    seen_pairs = set()
    duplicate_image_count = 0
    duplicate_pair_count = 0
    entity_counter = Counter()

    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            total += 1
            pair_key = (rec["image_path"], rec["caption"])

            if rec["image_path"] in seen_images:
                duplicate_image_count += 1
            seen_images.add(rec["image_path"])

            if pair_key in seen_pairs:
                duplicate_pair_count += 1
            seen_pairs.add(pair_key)

            for phrase in extract_entity_phrases(rec["caption"]):
                entity_counter[phrase] += 1

    n_entities = len(entity_counter)
    once = sum(1 for c in entity_counter.values() if c == 1)
    twice_or_more = n_entities - once

    print(f"Total records: {total}")
    print(f"Unique images: {len(seen_images)} ({total - len(seen_images)} repeat-image records, "
          f"{100*(total-len(seen_images))/total:.1f}%)")
    print(f"Exact duplicate (image, caption) pairs: {duplicate_pair_count} "
          f"({100*duplicate_pair_count/total:.1f}%)")
    print()
    print(f"Distinct entity phrases found: {n_entities}")
    print(f"  Appearing exactly once:  {once} ({100*once/n_entities:.1f}%)")
    print(f"  Appearing 2+ times:      {twice_or_more} ({100*twice_or_more/n_entities:.1f}%)")
    print()
    print("Top 15 most-repeated entity phrases (these are the ones the model")
    print("had a real chance to learn from repeated exposure):")
    for phrase, count in entity_counter.most_common(15):
        print(f"  {count:4d}  {phrase}")


if __name__ == "__main__":
    main()
