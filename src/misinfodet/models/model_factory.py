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
