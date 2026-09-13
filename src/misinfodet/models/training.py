"""Shared LoRA training loop for both instruction-tuning stages.

Uses a plain PyTorch loop rather than HF Trainer so the collation of LLaVA
conversation format (image tokens + masked prompt labels) is explicit and
auditable for the methodology chapter.
"""
from __future__ import annotations

import math
from pathlib import Path

from ..utils import Config, get_logger

log = get_logger(__name__)


class InstructionDataset:
    """Wraps prebuilt instruction examples ({image_path, prompt, target})."""

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def make_collate_fn(processor):
    """Collate examples into LLaVA inputs with prompt tokens masked out."""
    import torch
    from PIL import Image

    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"  # required for the prefix mask below

    def collate(batch: list[dict]):
        images, full_texts, prompt_lens = [], [], []
        for ex in batch:
            image = Image.open(ex["image_path"]).convert("RGB")
            images.append(image)
            conversation = [
                {"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": ex["prompt"]}]},
            ]
            prompt_text = processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            full_texts.append(prompt_text + ex["target"] + tokenizer.eos_token)
            # Real prompt length AFTER the processor expands the single
            # <image> placeholder into its patch tokens (576 for LLaVA-1.5
            # at 336px). Tokenizing prompt_text with the plain tokenizer
            # would count <image> as one token and undercount by ~575,
            # leaving image/prompt tokens in the loss.
            prompt_lens.append(
                processor(images=image, text=prompt_text, return_tensors="pt")
                ["input_ids"].shape[1]
            )

        inputs = processor(
            images=images, text=full_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=1024,
        )
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100
        inputs["labels"] = labels
        return inputs

    return collate


def train(model, processor, examples: list[dict], cfg: Config, save_dir: str | Path) -> None:
    import torch
    from torch.utils.data import DataLoader

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = InstructionDataset(examples)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=make_collate_fn(processor),
    )

    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.lr
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=max(total_steps, 1),
        pct_start=cfg.warmup_ratio,
    )

    wandb_run = None
    if cfg.wandb:
        try:
            import wandb

            wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.run_name,
                                   config=vars(cfg))
        except Exception as e:
            log.warning("wandb init failed (%s); continuing without it", e)

    model.train()
    global_step = 0
    for epoch in range(cfg.epochs):
        running = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss / cfg.grad_accum
            loss.backward()
            running += loss.item()
            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 10 == 0:
                    avg = running / (10 * 1)
                    log.info("epoch %d step %d loss %.4f", epoch, global_step, avg)
                    if wandb_run:
                        wandb_run.log({"loss": avg, "epoch": epoch}, step=global_step)
                    running = 0.0
                # Periodic checkpoint — a multi-hour unattended run (session
                # drop, accelerator change, timeout) loses only progress
                # since the LAST of these, not everything back to step 0.
                # Overwrites in place: LoRA adapters are small, and we only
                # need the most recent one, not a history of all of them.
                if cfg.save_every_steps and global_step % cfg.save_every_steps == 0:
                    model.save_pretrained(save_dir)
                    processor.save_pretrained(save_dir)
                    log.info(
                        "Checkpoint saved at step %d (epoch %d) to %s",
                        global_step, epoch, save_dir,
                    )

    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    log.info("Adapter saved to %s", save_dir)
    if wandb_run:
        wandb_run.finish()