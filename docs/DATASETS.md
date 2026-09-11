# Dataset acquisition

## COSMOS (primary, as of the dataset switch approved by Dr Olukanmi)

Access is gated behind a form: fill it out at
https://docs.google.com/forms/d/13kJQ2wlv7sxyXoaM1Ddon6Nq7dUJY_oftl-6xzwTGow —
after submitting, the confirmation page links directly to four files (no
separate approval wait):

- `cosmos_anns.zip` (~108MB) — annotations for all three splits
- `images_train.zip` (~18.6GB)
- `images_val.zip` (~4.4GB)
- `images_test.zip` (~312MB)

Extract so this exists:
```
data/cosmos/train_data.json
data/cosmos/val_data.json
data/cosmos/test_data.json
data/cosmos/train/...   (images)
data/cosmos/val/...     (images)
data/cosmos/test/...    (images)
```
Then run `python scripts/prepare_cosmos.py`.

**Known limitation, already discussed with and approved by Dr Olukanmi:**
only `test_data.json` (~1,700 images) carries an out-of-context label —
COSMOS's own method is self-supervised, so `train_data.json`/`val_data.json`
(160K+ images) have none. Stage 1 (news-domain entity alignment) uses the
unlabeled train/val data fine; Stage 2 (OOC fine-tuning) and evaluation are
constrained to the ~1,700 labeled test images, split further into
train/val/test for that stage — this is a real reduction from
NewsCLIPpings' 35,536 labeled training examples, and it also means SNIFFER/
LEMMA's reported numbers (on NewsCLIPpings) are no longer a direct,
apples-to-apples baseline comparison.

The test set's shape also differs from NewsCLIPpings/MMFakeBench: each
image comes with **two** candidate captions and one label for whether
pairing them together is out-of-context, not one caption with its own
label. See `src/misinfodet/data/cosmos.py` docstring for the exact schema
this produces, and note that consuming it needs a two-caption prompt
template, not the existing single-caption one.

## NewsCLIPpings (superseded — kept for reference only)

VisualNews' image archive (`origin.tar`, 91GB) proved impractical to
download reliably (multiple corruption incidents over several days); the
project switched to COSMOS above. This section is left for reference in
case NewsCLIPpings is revisited later.

NewsCLIPpings distributes only annotation files; images and captions come from
VisualNews, which is gated behind an access request.

1. **VisualNews**: open https://github.com/FuxiaoLiu/VisualNews-Repository and
   complete the data request form linked in its README. You will receive
   `origin.tar` (images) and `data.json` (metadata). Extract to:
   ```
   data/visualnews/origin/...
   data/visualnews/data.json
   ```
2. **NewsCLIPpings annotations**: clone https://github.com/g-luo/news_clippings
   and follow its download instructions (Google Drive links in the repo).
   Place so this path exists:
   ```
   data/newsclippings/news_clippings/data/merged_balanced/{train,val,test}.json
   ```
3. Run `python scripts/prepare_newsclippings.py`.

Merged/Balanced sizes should come out at roughly 71k train / 7k val / 7k test
pairs (train 35,536 unique falsified + matched counterparts). Verify counts
against the paper before training and record them in EXPERIMENT_LOG.md.

## MMFakeBench (secondary, evaluation only)

Download from the official release: https://github.com/liuxuannan/MMFakeBench.
Access is gated behind a Data Usage Protocol on the HuggingFace dataset page
(https://huggingface.co/datasets/liuxuannan/MMFakeBench) — request access there
first. Place so these exist:
```
data/mmfakebench/MMFakeBench_test.json
data/mmfakebench/MMFakeBench_test/...
```
Then run `python scripts/prepare_mmfakebench.py`. Never train on this set.

## Evidence corpus (optional, for offline/reproducible retrieval)

For fully reproducible ablations, snapshot web evidence once into
`data/evidence_corpus.jsonl` (one `{"title", "snippet", "url"}` per line) and
set `search_provider: offline`. A snapshot can be built by running the live
retriever once and saving every returned snippet.

## Search API keys

Live retrieval needs one of:
- Serper (https://serper.dev): `export SEARCH_API_KEY=...` with `search_provider: serper`
- Bing Web Search: same env var with `search_provider: bing`

Keys are read from the environment only. Never commit keys.
