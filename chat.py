#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-3B (Instruct) terminal chat with KV caching
---------------------------------------------------
- Uses Hugging Face Transformers' new Cache API (DynamicCache) for multi-turn reuse.
- Does *incremental* prefill of only the new tokens (delta) per turn and then
  streams token-by-token decoding — no re-prefill of the entire history.
- Works fully offline once model is downloaded.

Features
- Streaming output in terminal
- Slash commands:
  /help                show help
  /clear               clear conversation (keep system prompt)
  /system <text>       set system prompt (resets cache)
  /reset               reset cache (keeps history; next turn will re-prefill delta)
  /max <n>             set max_new_tokens
  /temp <float>        set temperature (0.0 = greedy)
  /topp <float>        set top_p
  /quit                exit
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.cache_utils import DynamicCache

# ----------------------------- Utils -----------------------------

def softmax(x):
    x = x - x.max(dim=-1, keepdim=True).values
    return torch.exp(x) / torch.exp(x).sum(dim=-1, keepdim=True)

def top_p_filtering(logits, top_p: float = 0.9, min_tokens_to_keep: int = 1):
    """Nucleus (top-p) filtering: keep the top tokens with cumulative probability >= top_p."""
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_mask = cumulative_probs <= top_p
    # Always keep at least min_tokens_to_keep
    sorted_mask[..., :min_tokens_to_keep] = True

    # Re-map to original indices
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
    # Set filtered tokens to -inf
    logits[~mask] = -float("inf")
    return logits

def printable_delta_decode(tok, generated_ids, last_printed_text):
    """Decode the full generated text and print only the new delta (for pretty streaming)."""
    text = tok.decode(generated_ids[0], skip_special_tokens=True)
    delta = text[len(last_printed_text):]
    return text, delta

def supports_token(tok, token_str: str) -> Optional[int]:
    tid = tok.convert_tokens_to_ids(token_str)
    return None if (tid is None or tid == tok.unk_token_id) else tid

# ------------------------- Chat Engine ---------------------------

class KVChatEngine:
    def __init__(self, model_id: str, system_prompt: str, device_map: str, dtype: str,
                 attn_impl: Optional[str], quant: Optional[str],
                 max_new_tokens: int, temperature: float, top_p: float):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        quant_cfg = None
        if quant in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            if quant == "4bit":
                quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            else:
                quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

        kwargs = dict(
            device_map=device_map,
            torch_dtype=(torch.bfloat16 if dtype == "bfloat16" else torch.float16) if dtype != "auto" else "auto",
        )
        if quant_cfg is not None:
            kwargs["quantization_config"] = quant_cfg
        if attn_impl is not None:
            kwargs["attn_implementation"] = attn_impl

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        torch.set_grad_enabled(False)

        # Chat state
        self.messages: List[dict] = [{"role": "system", "content": system_prompt}]
        self.cache = DynamicCache()
        self.cached_len = 0  # number of tokens already inside cache
        self.im_end_id = supports_token(self.tok, "<|im_end|>")
        self.eos_id = self.tok.eos_token_id

    def _encode_full(self, add_gen_prompt=True):
        """Encode the *entire* conversation into input_ids (but we will only prefill the delta)."""
        enc = self.tok.apply_chat_template(
            self.messages,
            add_generation_prompt=add_gen_prompt,
            return_tensors="pt",
            return_dict=True
        )
        return {k: v.to(self.model.device) for k, v in enc.items()}

    def _prefill_delta(self, full_input_ids: torch.Tensor):
        """Prefill only the delta (new tokens beyond self.cached_len) into the cache.
        Returns the model outputs from the prefill forward (to get logits for the
        last prompt token), or None if nothing was prefetched.
        """
        total_len = full_input_ids.shape[1]
        if self.cached_len == total_len:
            return None  # nothing to add

        if self.cached_len > total_len:
            # History changed (e.g., /system or /clear). Reset cache completely.
            self.cache = DynamicCache()
            self.cached_len = 0

        delta_ids = full_input_ids[:, self.cached_len:]
        if delta_ids.numel() == 0:
            return None

        # Build attention_mask for (cached_len + delta_len)
        new_total = self.cached_len + delta_ids.shape[1]
        attn_mask = torch.ones((1, new_total), dtype=torch.long, device=full_input_ids.device)

        cache_position = torch.arange(self.cached_len, new_total, dtype=torch.int64, device=full_input_ids.device)

        outputs = self.model(
            input_ids=delta_ids,
            attention_mask=attn_mask,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=cache_position,
        )
        self.cached_len = new_total  # we've extended cache to cover full_input_ids
        return outputs

    def _sample_next_token(self, logits: torch.Tensor, ban_stop: bool = False):
        """Temperature + top-p sampling; if temperature <= 0, do greedy.
        If ban_stop is True, disallow sampling EOS and <|im_end|>.
        """
        last_logits = logits[:, -1, :].clone()

        # Optionally ban stop tokens for robustness (helps avoid empty replies)
        if ban_stop:
            if self.eos_id is not None and self.eos_id >= 0:
                last_logits[:, self.eos_id] = -float("inf")
            if self.im_end_id is not None and self.im_end_id >= 0:
                last_logits[:, self.im_end_id] = -float("inf")

        if self.temperature <= 0.0:
            next_id = torch.argmax(last_logits, dim=-1, keepdim=True)
            return next_id

        scaled = last_logits / max(self.temperature, 1e-6)
        if 0.0 < self.top_p < 1.0:
            scaled = top_p_filtering(scaled, top_p=self.top_p, min_tokens_to_keep=1)
        probs = torch.softmax(scaled, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        return next_id

    def generate_stream(self):
        """Run one assistant turn: prefill delta, then decode tokens, streaming to stdout."""
        # 1) Encode full conversation (with assistant role open)
        enc = self._encode_full(add_gen_prompt=True)
        full_ids = enc["input_ids"]

        # 2) Prefill only the new tokens into the cache and get logits at the last prompt token
        prefill_out = self._prefill_delta(full_ids)

        # 3) Now decode token-by-token starting from the current cache length
        attn_len = self.cached_len
        attn_mask = torch.ones((1, attn_len), dtype=torch.long, device=full_ids.device)
        generated = []
        last_printed = ""

        # First step logits come from prefill; if for some reason there was no new delta,
        # fall back to getting logits at the last cached position (only if cached_len > 0).
        current_logits = None
        if prefill_out is not None and hasattr(prefill_out, "logits"):
            current_logits = prefill_out.logits
        elif attn_len > 0:
            # Fallback: query logits at the last cached position.
            last_token = full_ids[:, attn_len - 1 : attn_len]
            out = self.model(
                input_ids=last_token,
                attention_mask=attn_mask,
                past_key_values=self.cache,
                use_cache=True,
                cache_position=torch.tensor([attn_len - 1], dtype=torch.int64, device=full_ids.device),
            )
            current_logits = out.logits

        stop_reached = False
        first_step = True
        for _ in range(self.max_new_tokens):
            # Sample next token from current logits (which correspond to the last context position)
            if current_logits is None:
                # Nothing to sample from; break defensively
                break
            next_id = self._sample_next_token(current_logits, ban_stop=first_step)

            # Stop if eos or <|im_end|>
            ni = int(next_id[0,0].item())
            if (self.eos_id is not None and ni == self.eos_id) or (self.im_end_id is not None and ni == self.im_end_id):
                stop_reached = True

            # Append to running text (pretty-print only new delta each loop)
            generated.append(next_id)
            full_text, delta = printable_delta_decode(self.tok, torch.cat(generated, dim=1), last_printed)
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
                last_printed = full_text

            # Prepare next step: advance mask/position and extend cache with this new token
            attn_len += 1
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1,1))], dim=-1)
            cache_pos = torch.tensor([attn_len - 1], dtype=torch.int64, device=full_ids.device)

            out = self.model(
                input_ids=next_id,
                attention_mask=attn_mask,
                past_key_values=self.cache,
                use_cache=True,
                cache_position=cache_pos,
            )
            current_logits = out.logits
            first_step = False

            if stop_reached:
                break

        print()  # newline after streaming
        return last_printed  # the full decoded assistant content

    # ---------------------- Public API ----------------------

    def ask(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        reply = self.generate_stream()
        self.messages.append({"role": "assistant", "content": reply})
        # Re-sync cache to canonical re-encoding of full history to avoid
        # any decode->encode tokenization drift across turns.
        try:
            enc_full = self.tok.apply_chat_template(
                self.messages,
                add_generation_prompt=False,
                return_tensors="pt",
                return_dict=True,
            )
            enc_full = {k: v.to(self.model.device) for k, v in enc_full.items()}
            full_ids = enc_full["input_ids"]
            self.cache = DynamicCache()
            self.cached_len = 0
            if full_ids.numel() > 0:
                attn_mask = torch.ones((1, full_ids.shape[1]), dtype=torch.long, device=full_ids.device)
                cache_pos = torch.arange(0, full_ids.shape[1], dtype=torch.int64, device=full_ids.device)
                _ = self.model(
                    input_ids=full_ids,
                    attention_mask=attn_mask,
                    past_key_values=self.cache,
                    use_cache=True,
                    cache_position=cache_pos,
                )
                self.cached_len = full_ids.shape[1]
        except Exception:
            # If re-sync fails for any reason, just continue without breaking chat.
            pass
        return reply

    def set_system(self, text: str):
        self.messages[0] = {"role": "system", "content": text}
        self.cache = DynamicCache()
        self.cached_len = 0

    def clear(self):
        sys_prompt = self.messages[0]
        self.messages = [sys_prompt]
        self.cache = DynamicCache()
        self.cached_len = 0

    def reset_cache(self):
        self.cache = DynamicCache()
        self.cached_len = 0

# ----------------------------- Main ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Terminal chat with Qwen2.5-3B and KV cache (incremental).")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="HF model id")
    parser.add_argument("--device-map", default="auto", help="transformers device_map")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16"], help="compute dtype")
    parser.add_argument("--attn", default=None, choices=[None, "flash_attention_2"], help="attention backend")
    parser.add_argument("--quant", default=None, choices=[None, "4bit", "8bit"], help="load quantized weights")
    parser.add_argument("--system", default="You are a helpful assistant.", help="system prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="max new tokens per turn")
    parser.add_argument("--temperature", type=float, default=0.7, help="sampling temperature (0 = greedy)")
    parser.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling top_p")
    args = parser.parse_args()

    engine = KVChatEngine(
        model_id=args.model,
        system_prompt=args.system,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_impl=args.attn,
        quant=args.quant,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\nQwen2.5-3B Terminal Chat (KV cache, incremental)\n"
          "Model: {}\nSystem: {}\n".format(args.model, args.system))
    print("Commands: /help  /clear  /system <text>  /reset  /max <n>  /temp <f>  /topp <f>  /quit\n")

    try:
        while True:
            try:
                user = input("\n> ").strip()
            except EOFError:
                print()
                break

            if not user:
                continue

            if user.startswith("/"):
                cmd, *rest = user[1:].split(" ", 1)
                arg = rest[0] if rest else ""
                if cmd == "help":
                    print("Commands:\n"
                          "  /help               Show this help\n"
                          "  /clear              Clear conversation (keep system prompt)\n"
                          "  /system <text>      Set system prompt (resets cache)\n"
                          "  /reset              Reset KV cache (keeps history)\n"
                          "  /max <n>            Set max_new_tokens\n"
                          "  /temp <f>           Set temperature (0.0 = greedy)\n"
                          "  /topp <f>           Set top_p\n"
                          "  /quit               Exit")
                    continue
                elif cmd == "clear":
                    engine.clear()
                    print("History cleared.")
                    continue
                elif cmd == "system":
                    if arg:
                        engine.set_system(arg)
                        print("System prompt updated and cache reset.")
                    else:
                        print("Usage: /system <text>")
                    continue
                elif cmd == "reset":
                    engine.reset_cache()
                    print("KV cache reset.")
                    continue
                elif cmd == "max":
                    try:
                        engine.max_new_tokens = max(1, int(arg))
                        print(f"max_new_tokens = {engine.max_new_tokens}")
                    except Exception:
                        print("Usage: /max <int>")
                    continue
                elif cmd == "temp":
                    try:
                        engine.temperature = float(arg)
                        print(f"temperature = {engine.temperature}")
                    except Exception:
                        print("Usage: /temp <float>")
                    continue
                elif cmd == "topp":
                    try:
                        engine.top_p = float(arg)
                        print(f"top_p = {engine.top_p}")
                    except Exception:
                        print("Usage: /topp <float>")
                    continue
                elif cmd == "quit":
                    break
                else:
                    print("Unknown command. Type /help")
                    continue

            # Regular user turn
            engine.ask(user)

    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")

if __name__ == "__main__":
    main()
