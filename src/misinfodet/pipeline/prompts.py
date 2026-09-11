"""All prompt templates for the system, centralised for auditability.

Keeping every prompt here makes the methodology chapter easy to write and the
system easy to ablate: any prompt change is a single, version-controlled diff.
"""

# --------------------------------------------------------------------------- #
# Stage 1: news-domain entity alignment
# --------------------------------------------------------------------------- #
STAGE1_ALIGNMENT_PROMPTS = [
    "Describe this news photograph, naming the specific people, places, and events shown.",
    "Write a news caption for this image, identifying the named entities involved.",
    "What is happening in this news image? Identify who is pictured and where.",
]

# --------------------------------------------------------------------------- #
# Stage 2 / inference: internal consistency verdict
# --------------------------------------------------------------------------- #
VERDICT_PROMPT = (
    "You are a fact-checking assistant analysing a news image and its caption.\n"
    'Caption: "{caption}"\n\n'
    "Assess whether the caption genuinely describes this image, or whether this "
    "is an out-of-context pair (a real image paired with an unrelated or "
    "misleading caption). Consider the people, places, events, and time period "
    "visible in the image versus those claimed in the caption.\n\n"
    "Answer in exactly this format:\n"
    "VERDICT: <consistent | out-of-context>\n"
    "REASONING: <2-4 sentences explaining the key visual and textual evidence>"
)

STAGE2_TARGET_REAL = (
    "VERDICT: consistent\n"
    "REASONING: The entities and scene visible in the image correspond to those "
    "described in the caption, and no cross-modal contradiction is present."
)

STAGE2_TARGET_FAKE = (
    "VERDICT: out-of-context\n"
    "REASONING: The people, place, or event shown in the image do not match "
    "what the caption claims, indicating the image has been repurposed to "
    "support an unrelated statement."
)

# --------------------------------------------------------------------------- #
# Stage 2 / inference: paired-caption consistency (COSMOS test schema — one
# image, two candidate captions, single label for whether pairing them is
# out-of-context; NOT the same task shape as the single-caption prompts above)
# --------------------------------------------------------------------------- #
PAIRED_VERDICT_PROMPT = (
    "You are a fact-checking assistant analysing a news image against two "
    "candidate captions that have each been associated with it.\n"
    'Caption 1: "{caption1}"\n'
    'Caption 2: "{caption2}"\n\n'
    "Assess whether using both captions together with this image constitutes "
    "an out-of-context pairing — i.e. whether the two captions describe "
    "genuinely different people, places, or events, such that at most one "
    "can truthfully apply to this image. Consider the entities, setting, and "
    "time period visible in the image versus those in each caption.\n\n"
    "Answer in exactly this format:\n"
    "VERDICT: <consistent | out-of-context>\n"
    "REASONING: <2-4 sentences explaining the key visual and textual evidence>"
)

PAIRED_FINAL_VERDICT_PROMPT = (
    "You are a fact-checking assistant analysing a news image against two "
    "candidate captions that have each been associated with it.\n"
    'Caption 1: "{caption1}"\n'
    'Caption 2: "{caption2}"\n\n'
    "Your initial visual assessment was:\n{initial_reasoning}\n\n"
    "External evidence retrieved from the web:\n{evidence}\n\n"
    "Using both the image and the external evidence, decide whether pairing "
    "these two captions with this image is out-of-context.\n\n"
    "Answer in exactly this format:\n"
    "VERDICT: <consistent | out-of-context>\n"
    "CONFIDENCE: <low | medium | high>\n"
    "REASONING: <3-6 sentences citing the specific visual and external evidence "
    "that determined your verdict>"
)

STAGE2_PAIRED_TARGET_REAL = (
    "VERDICT: consistent\n"
    "REASONING: Both captions are compatible with the entities and scene "
    "visible in the image, and no cross-modal contradiction is present."
)

STAGE2_PAIRED_TARGET_FAKE = (
    "VERDICT: out-of-context\n"
    "REASONING: The two captions describe different people, places, or "
    "events, so at most one can genuinely correspond to this image — "
    "indicating an out-of-context pairing."
)

# --------------------------------------------------------------------------- #
# Retrieval: targeted query generation (LEMMA-style)
# --------------------------------------------------------------------------- #
QUERY_GENERATION_PROMPT = (
    "You are verifying a news caption against its image.\n"
    'Caption: "{caption}"\n'
    "Your initial assessment: {initial_reasoning}\n\n"
    "List the {n_queries} most useful web search queries to verify or refute "
    "the specific factual claims in this caption (named people, places, dates, "
    "events). One query per line, no numbering, no commentary."
)

# --------------------------------------------------------------------------- #
# Retrieval: evidence re-ranking (LVLM4FV-style)
# --------------------------------------------------------------------------- #
RERANK_PROMPT = (
    'Claim being verified: "{caption}"\n\n'
    "Candidate evidence snippet:\n{snippet}\n\n"
    "Rate how useful this snippet is for verifying or refuting the claim, on a "
    "scale of 0 (irrelevant) to 10 (directly probative). Answer with only the number."
)

# --------------------------------------------------------------------------- #
# Final verdict with retrieved evidence
# --------------------------------------------------------------------------- #
FINAL_VERDICT_PROMPT = (
    "You are a fact-checking assistant analysing a news image and its caption.\n"
    'Caption: "{caption}"\n\n'
    "Your initial visual assessment was:\n{initial_reasoning}\n\n"
    "External evidence retrieved from the web:\n{evidence}\n\n"
    "Using both the image and the external evidence, decide whether the caption "
    "genuinely describes this image or is an out-of-context pairing.\n\n"
    "Answer in exactly this format:\n"
    "VERDICT: <consistent | out-of-context>\n"
    "CONFIDENCE: <low | medium | high>\n"
    "REASONING: <3-6 sentences citing the specific visual and external evidence "
    "that determined your verdict>"
)

# --------------------------------------------------------------------------- #
# Explanation quality: LLM-as-judge rubric
# --------------------------------------------------------------------------- #
EXPLANATION_JUDGE_PROMPT = (
    "You are grading the quality of a fact-checking explanation.\n\n"
    'Caption under review: "{caption}"\n'
    "Ground-truth label: {gold_label}\n"
    "System verdict: {pred_label}\n"
    "System explanation:\n{explanation}\n\n"
    "Score the explanation on three criteria, each 1-5:\n"
    "- factual_accuracy: are the claims in the explanation correct and non-hallucinated?\n"
    "- coherence: is the reasoning logically structured and internally consistent?\n"
    "- relevance: does it address the actual image-caption relationship rather than generic statements?\n\n"
    'Respond with only a JSON object: {{"factual_accuracy": n, "coherence": n, "relevance": n}}'
)
