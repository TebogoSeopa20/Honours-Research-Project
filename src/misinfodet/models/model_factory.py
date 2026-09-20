"""LLaVA loading with optional 4-bit quantisation and LoRA adapters.

Heavy imports (torch/transformers) are kept inside functions so the package
imports cleanly on CPU-only machines for tests and data preparation.
"""
from __future__ import annotations

from ..utils import Config, get_logger

log = get_logger(__name__)


def load_llava(cfg: Config, trainable: bool = False):
    """Load LLaVA base model (+ optional adapters) and its processor.

    Returns (model, processor). Applies, in order:
      1. 4-bit quantisation (QLoRA-compatible) if cfg.load_in_4bit
      2. Stage 1 adapter if cfg.stage1_adapter is set
      3. Stage 2 adapter if cfg.stage2_adapter is set
      4. A fresh trainable LoRA if trainable=True
    """
    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        LlavaForConditionalGeneration,
    )

    quant_config = None
    if cfg.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    log.info("Loading base model %s (4bit=%s)", cfg.base_model, cfg.load_in_4bit)
    model = LlavaForConditionalGeneration.from_pretrained(
        cfg.base_model,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(cfg.base_model)

    from peft import PeftModel

    for name, adapter in (("stage1", cfg.stage1_adapter), ("stage2", cfg.stage2_adapter)):
        if adapter:
            log.info("Applying %s adapter from %s", name, adapter)
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload() if not trainable else model

    if trainable:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    return model, processor


def generate(model, processor, image, prompt: str, cfg: Config) -> str:
    """Single-example generation with the LLaVA chat template."""
    import torch

    conversation = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=max(cfg.temperature, 1e-5),
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def generate_text_only(model, processor, prompt: str, cfg: Config) -> str:
    """Text-only generation (used for query generation and re-ranking)."""
    import torch

    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=128, do_sample=False
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


def debug_verdict_tokenization(model, processor, image, prompt: str, cfg: Config) -> None:
    """Run this FIRST, on one real example, before trusting calibration.

    CORRECTED: earlier version forced "VERDICT:" as a prefix, assuming the
    model continues the literal training-target text. Confirmed against
    real generation that this is wrong — actual outputs are bare verdict
    words ('Out-of-context', 'Consistent.') with no "VERDICT:" prefix at
    all. This checks the genuinely first free-generation token instead.
    """
    import torch

    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    next_token_logits = outputs.logits[0, -1, :]
    top5 = torch.topk(next_token_logits, 5)
    print("Top-5 real FIRST-token predictions (no forced prefix):")
    for logit, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  {processor.tokenizer.decode([idx])!r}  (logit={logit:.2f})")

    p_out, p_consistent = _bucket_probs(next_token_logits, processor.tokenizer)
    print(f"\nSummed P(out-of-context)={p_out:.4f}  P(consistent)={p_consistent:.4f}  "
          f"(normalized score={p_out / (p_out + p_consistent + 1e-9):.4f})")


def _bucket_probs(next_token_logits, tokenizer, top_k: int = 200) -> tuple[float, float]:
    """Sums softmax probability across the top_k most likely next tokens,
    bucketed by whether their decoded text (stripped, lowercased) starts
    with "out" or "cons" — robust to both capitalization variance and
    tokenizer-specific leading-space artifacts, unlike comparing two fixed
    token ids directly (confirmed necessary against real model output)."""
    import torch

    top = torch.topk(next_token_logits, top_k)
    probs = torch.softmax(top.values, dim=0)
    p_out, p_consistent = 0.0, 0.0
    for prob, idx in zip(probs.tolist(), top.indices.tolist()):
        text = tokenizer.decode([idx]).strip().lower()
        if text.startswith("out"):
            p_out += prob
        elif text.startswith("cons"):
            p_consistent += prob
    return p_out, p_consistent


def get_verdict_score(model, processor, image, prompt: str, cfg: Config) -> float:
    """Returns P(out-of-context) as a continuous score in [0, 1].

    CORRECTED AGAIN: confirmed via diagnostic (25 real examples) that the
    model's actual behavior is MIXED — some outputs jump straight to the
    verdict word, but 20% first generate "VERDICT:" before it. A single
    forward pass checking only the very first token misses that 20%
    entirely. This version does a real short generation (cheap — 10
    tokens, not the full 320) and scans the ACTUAL generated tokens for
    wherever the real class decision happens, using the same greedy
    decoding mechanism as real generation, so it can't systematically
    diverge from it the way a single-forward-pass guess can.
    """
    import torch

    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen_out = model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
        )

    for step_logits in gen_out.scores:
        token_id = int(step_logits[0].argmax())
        text = processor.tokenizer.decode([token_id]).strip().lower()
        if text.startswith("out") or text.startswith("cons"):
            p_out, p_consistent = _bucket_probs(step_logits[0], processor.tokenizer)
            return p_out / (p_out + p_consistent + 1e-9)
    return 0.5  # no clear decision token found in the first 10 — neutral fallback