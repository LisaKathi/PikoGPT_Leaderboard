from __future__ import annotations

import sys
import warnings

import torch
from transformers import GPT2TokenizerFast

from src.config.settings import InferenceConfig, ModelConfig
from src.model.gpt import GPT
from src.utils.device import get_device


_ALPACA_WRAPPER = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n{body}\n\n### Response:\n"
)


def _wrap_leaderboard_prompt(prompt: str) -> str:
    """
    Wrap a raw leaderboard MC prompt in the Alpaca preamble used during SFT,
    so the model recognises it as a multiple-choice task.
    Non-MC prompts (LAMBADA continuations) pass through unchanged.
    """
    if not prompt.rstrip().endswith("Answer:"):
        return prompt

    body_full = prompt.rstrip()
    body_full = body_full[: -len("Answer:")].rstrip()

    if body_full.startswith("Question:"):
        # OpenBookQA / ARC-style
        instruction = (
            "Answer the following multiple-choice question with the letter of the correct answer."
        )
        body = body_full[len("Question:") :].strip()
    elif body_full.startswith("Context:"):
        # HellaSwag / WinoGrande-style
        instruction = "Choose the most logical continuation."
        body = body_full[len("Context:") :].strip()
    else:
        return prompt

    return _ALPACA_WRAPPER.format(instruction=instruction, body=body)


class InferenceRunner:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config

    def run(self) -> str:
        cfg = self.config
        if cfg.checkpoint is None:
            raise ValueError("--checkpoint is required for inference")

        torch.manual_seed(cfg.seed)
        device = get_device(cfg.device)
        ckpt = torch.load(cfg.checkpoint, map_location=device)
        model_cfg = ModelConfig(**ckpt["model_config"])
        model = GPT(model_cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)

        prompt_text = _wrap_leaderboard_prompt(cfg.prompt) if cfg.leaderboard else cfg.prompt
        is_lambada = cfg.leaderboard and not cfg.prompt.rstrip().endswith("Answer:")
        # Strip trailing whitespace for LAMBADA so the model emits " word" tokens
        # (with leading space) instead of subword fragments like "vernal".
        if is_lambada:
            prompt_text = prompt_text.rstrip()
        input_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
        eos_token_id = 50256
        # GPT-2 BPE: 198='\n', 628='\n\n', 220=' ', 50256=EOS.
        # Block newlines + EOS for the first generated token; allow space and other
        # leading characters so word continuations like " signs" can still emerge.
        LAMBADA_BLOCK_FIRST = (198, 628, 50256)

        if cfg.temperature == 0.0:
            with torch.no_grad():
                for step in range(cfg.max_tokens):
                    idx_cond = input_ids[:, -model_cfg.context_len:]
                    logits, _ = model(idx_cond)
                    last_logits = logits[:, -1, :]
                    if is_lambada and step == 0:
                        last_logits = last_logits.clone()
                        for tid in LAMBADA_BLOCK_FIRST:
                            last_logits[:, tid] = float("-inf")
                    next_token = last_logits.argmax(dim=-1, keepdim=True)
                    input_ids = torch.cat((input_ids, next_token), dim=1)
                    if next_token.item() == eos_token_id:
                        break
        else:
            input_ids = model.generate(
                input_ids,
                max_new_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )

        prompt_len = len(tokenizer.encode(prompt_text))
        generated_ids = input_ids[0][prompt_len:].tolist()
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Defensive cleanup for LAMBADA: strip leading newlines / quotes so the
        # leaderboard's first-word extractor lands on a real candidate.
        if is_lambada:
            output_text = output_text.lstrip(" \n\r\t\"'“”‘’")

        if cfg.leaderboard:
            enc = sys.stdout.encoding or "utf-8"
            safe = output_text.encode(enc, errors="replace").decode(enc)
            sys.stdout.write(safe)
            sys.stdout.flush()
        else:
            print(output_text)
        return output_text
