#!/usr/bin/env python3
from __future__ import annotations

from stackelberg_codepo.modeling.chat_template import safe_apply_chat_template

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def convert_messages(conversations: list[dict[str, str]]) -> list[dict[str, str]]:
    role_map = {"system": "system", "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}
    messages = []
    for item in conversations:
        role = role_map.get(item.get("from", "user"), item.get("role", "user"))
        content = item.get("value", item.get("content", ""))
        messages.append({"role": role, "content": content})
    return messages


def response_text(item: dict[str, Any], key: str) -> str:
    value = item[key]
    if isinstance(value, dict):
        return str(value.get("value", ""))
    return str(value)


def build_sequence(tokenizer, item: dict[str, Any], response_key: str, max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    messages = convert_messages(item["conversations"])
    prompt = safe_apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    resp = response_text(item, response_key)
    response_ids = tokenizer.encode(resp, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
        response_ids.append(eos_id)

    if len(response_ids) >= max_length:
        response_ids = response_ids[: max_length - 1]
        if eos_id is not None:
            response_ids.append(eos_id)
    max_prompt_len = max(1, max_length - len(response_ids))
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "response_len": torch.tensor(len(response_ids), dtype=torch.long, device=device),
        "total_len": torch.tensor(len(input_ids), dtype=torch.long, device=device),
    }


def collate_sequences(sequences: list[dict[str, torch.Tensor]], pad_token_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    max_len = max(int(seq["total_len"].detach().cpu().item()) for seq in sequences)
    input_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    for seq in sequences:
        pad_len = max_len - int(seq["total_len"].detach().cpu().item())
        input_rows.append(F.pad(seq["input_ids"], (0, pad_len), value=pad_token_id))
        label_rows.append(F.pad(seq["labels"], (0, pad_len), value=-100))
        mask_rows.append(F.pad(seq["attention_mask"], (0, pad_len), value=0))
    return {
        "input_ids": torch.stack(input_rows).to(device),
        "labels": torch.stack(label_rows).to(device),
        "attention_mask": torch.stack(mask_rows).to(device),
        "response_len": torch.stack([seq["response_len"] for seq in sequences]).to(device),
        "total_len": torch.stack([seq["total_len"] for seq in sequences]).to(device),
    }


def sequence_logprob(model, batch: dict[str, torch.Tensor], normalize: bool) -> torch.Tensor:
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * mask
    summed = token_logps.sum(dim=-1)
    if normalize:
        denom = mask.sum(dim=-1).clamp_min(1)
        return summed / denom
    return summed


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    pi_logratios = policy_chosen - policy_rejected
    ref_logratios = ref_chosen - ref_rejected
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits), logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal weighted DPO smoke trainer for role preference JSONL.")
    parser.add_argument("--model-path", default="/workspace/models/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-path", default=None, help="Optional LoRA adapter used to warm-start the trainable policy.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--normalize-logprob", action="store_true", help="Use mean response log-prob instead of summed log-prob.")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load policy/reference base models with bitsandbytes 8-bit quantization.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load policy/reference base models with bitsandbytes 4-bit NF4 quantization.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    data = load_jsonl(Path(args.data))
    if not data:
        raise ValueError(f"No examples found: {args.data}")
    random.shuffle(data)
    data = data[: args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Use only one of --load-in-8bit or --load-in-4bit")
    quantization_config = None
    device_map = None
    if args.load_in_8bit or args.load_in_4bit:
        if device.type != "cuda":
            raise ValueError("k-bit loading requires a CUDA device")
        device_map = {"": device.index if device.index is not None else 0}
        if args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            event = "load_in_4bit_enabled"
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            event = "load_in_8bit_enabled"
        print(json.dumps({"event": event, "device_map": device_map}, ensure_ascii=False), flush=True)

    common_load_kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if quantization_config is not None:
        common_load_kwargs["quantization_config"] = quantization_config
        common_load_kwargs["device_map"] = device_map

    policy = AutoModelForCausalLM.from_pretrained(args.model_path, **common_load_kwargs)
    policy.config.use_cache = False
    if args.load_in_8bit or args.load_in_4bit:
        policy = prepare_model_for_kbit_training(policy)
    if args.adapter_path:
        adapter_path = Path(args.adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Warm-start adapter path does not exist: {adapter_path}")
        policy = PeftModel.from_pretrained(policy, adapter_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        policy = get_peft_model(policy, lora_config)
    if not (args.load_in_8bit or args.load_in_4bit):
        policy.to(device)
    policy.train()

    reference = AutoModelForCausalLM.from_pretrained(args.model_path, **common_load_kwargs)
    reference.config.use_cache = False
    if not (args.load_in_8bit or args.load_in_4bit):
        reference.to(device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "weighted_dpo_smoke_log.jsonl"

    logs: list[dict[str, Any]] = []
    batch_size = max(1, args.batch_size)
    grad_accum = max(1, args.gradient_accumulation_steps)
    for step in range(args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        step_rows: list[dict[str, Any]] = []
        total_weighted_loss = 0.0
        total_unweighted_loss = 0.0
        total_weight = 0.0
        total_items = 0
        last_logits: torch.Tensor | None = None
        for micro_step in range(grad_accum):
            start = (step * grad_accum + micro_step) * batch_size
            batch_items = [data[(start + i) % len(data)] for i in range(batch_size)]
            weights = [float(item.get("weight", 1.0)) for item in batch_items]
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
            chosen = collate_sequences(
                [build_sequence(tokenizer, item, "chosen", args.max_length, device) for item in batch_items],
                int(pad_token_id),
                device,
            )
            rejected = collate_sequences(
                [build_sequence(tokenizer, item, "rejected", args.max_length, device) for item in batch_items],
                int(pad_token_id),
                device,
            )

            policy_chosen = sequence_logprob(policy, chosen, args.normalize_logprob)
            policy_rejected = sequence_logprob(policy, rejected, args.normalize_logprob)
            with torch.no_grad():
                ref_chosen = sequence_logprob(reference, chosen, args.normalize_logprob)
                ref_rejected = sequence_logprob(reference, rejected, args.normalize_logprob)

            unweighted_loss, logits = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, args.beta)
            weighted_loss = (unweighted_loss * weight_tensor).mean()
            (weighted_loss / grad_accum).backward()
            last_logits = logits.detach()
            total_weighted_loss += float(weighted_loss.detach().cpu().item()) * len(batch_items)
            total_unweighted_loss += float(unweighted_loss.detach().mean().cpu().item()) * len(batch_items)
            total_weight += sum(weights)
            total_items += len(batch_items)
            for idx, item in enumerate(batch_items):
                step_rows.append({
                    "weight": weights[idx],
                    "unweighted_loss": float(unweighted_loss[idx].detach().cpu().item()),
                    "weighted_loss": float((unweighted_loss[idx] * weight_tensor[idx]).detach().cpu().item()),
                    "dpo_logit": float(logits[idx].detach().cpu().item()),
                    "policy_chosen_logp": float(policy_chosen[idx].detach().cpu().item()),
                    "policy_rejected_logp": float(policy_rejected[idx].detach().cpu().item()),
                    "ref_chosen_logp": float(ref_chosen[idx].detach().cpu().item()),
                    "ref_rejected_logp": float(ref_rejected[idx].detach().cpu().item()),
                    "chosen_response_len": int(chosen["response_len"][idx].detach().cpu().item()),
                    "rejected_response_len": int(rejected["response_len"][idx].detach().cpu().item()),
                    "chosen_total_len": int(chosen["total_len"][idx].detach().cpu().item()),
                    "rejected_total_len": int(rejected["total_len"][idx].detach().cpu().item()),
                    "metadata": item.get("metadata", {}),
                })
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        row = {
            "step": step + 1,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "weight": total_weight / max(total_items, 1),
            "unweighted_loss": total_unweighted_loss / max(total_items, 1),
            "weighted_loss": total_weighted_loss / max(total_items, 1),
            "dpo_logit": float(last_logits.mean().cpu().item()) if last_logits is not None else 0.0,
            "grad_norm": float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
            "examples": step_rows,
        }
        logs.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    with log_path.open("w", encoding="utf-8") as f:
        for row in logs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "data": args.data,
        "output_dir": str(output_dir),
        "num_loaded_examples": len(data),
        "max_steps": args.max_steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "normalize_logprob": args.normalize_logprob,
        "load_in_8bit": args.load_in_8bit,
        "load_in_4bit": args.load_in_4bit,
        "log_path": str(log_path),
        "mean_weight": sum(row["weight"] for row in logs) / len(logs) if logs else 0.0,
        "mean_unweighted_loss": sum(row["unweighted_loss"] for row in logs) / len(logs) if logs else 0.0,
        "mean_weighted_loss": sum(row["weighted_loss"] for row in logs) / len(logs) if logs else 0.0,
    }
    (output_dir / "weighted_dpo_smoke_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
