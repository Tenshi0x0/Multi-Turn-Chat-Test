#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-3B (Instruct) terminal chat using model.generate + KV cache
-------------------------------------------------------------------
- Keeps a DynamicCache across turns. Each turn:
  1) Prefill only the delta tokens of the new prompt into the cache
  2) Call `model.generate` seeded with the existing cache and the last prompt token
  3) Stream tokens via TextIteratorStreamer (optional)
"""
import argparse
import sys
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.cache_utils import DynamicCache


def supports_token(tok, token_str: str) -> Optional[int]:
    tid = tok.convert_tokens_to_ids(token_str)
    return None if (tid is None or tid == tok.unk_token_id) else tid


class GenerateChatEngine:
    def __init__(
        self,
        model_id: str,
        system_prompt: str,
        device_map: str,
        dtype: str,
        attn_impl: Optional[str],
        quant: Optional[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool = True,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stream = stream

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

        self.messages: List[dict] = [{"role": "system", "content": system_prompt}]
        self.im_end_id = supports_token(self.tok, "<|im_end|>")
        self.eos_id = self.tok.eos_token_id

        # KV cache state reused across turns
        self.cache = DynamicCache()
        self.cached_len = 0

        # Ensure pad token is defined for generate
        if self.tok.pad_token_id is None and self.eos_id is not None:
            self.tok.pad_token_id = self.eos_id

    def _encode_full(self, add_gen_prompt: bool = True):
        text = self.tok.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=add_gen_prompt,
        )
        return self.tok([text], return_tensors="pt").to(self.model.device)

    def _prefill_delta(self, full_input_ids: torch.Tensor):
        total_len = full_input_ids.shape[1]
        if self.cached_len == total_len:
            return None
        if self.cached_len > total_len:
            # History changed (e.g., system updated); reset cache
            self.cache = DynamicCache()
            self.cached_len = 0

        delta_ids = full_input_ids[:, self.cached_len:]
        if delta_ids.numel() == 0:
            return None

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
        self.cached_len = new_total
        return outputs

    def _eos_ids(self):
        ids = []
        if self.eos_id is not None:
            ids.append(int(self.eos_id))
        if self.im_end_id is not None:
            ids.append(int(self.im_end_id))
        if not ids:
            return None
        return ids[0] if len(ids) == 1 else ids

    def _do_generate(self) -> str:
        # 1) Encode full conversation with assistant generation prompt
        enc = self._encode_full(add_gen_prompt=True)
        full_ids = enc["input_ids"]

        # 2) Call generate with full_ids, letting HF Generation slice out the delta based on past_key_values
        eos_ids = self._eos_ids()
        do_sample = self.temperature is not None and self.temperature > 0.0

        gen_kwargs = dict(
            inputs=full_ids,
            past_key_values=self.cache,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            eos_token_id=eos_ids,
            pad_token_id=self.tok.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
        )
        if do_sample:
            gen_kwargs.update(
                temperature=max(self.temperature, 1e-6),
                top_p=self.top_p,
            )

        if self.stream:
            streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer

            import threading

            out_box = {"out": None}

            def _worker():
                out = self.model.generate(**gen_kwargs)
                out_box["out"] = out

            t = threading.Thread(target=_worker)
            t.start()

            collected = []
            for piece in streamer:
                sys.stdout.write(piece)
                sys.stdout.flush()
                collected.append(piece)
            sys.stdout.write("\n")
            sys.stdout.flush()
            t.join()

            out = out_box["out"]
            # Update cache and cached_len from generate outputs
            if hasattr(out, "past_key_values") and out.past_key_values is not None:
                self.cache = out.past_key_values
                try:
                    self.cached_len = self.cache.get_seq_length()
                except Exception:
                    # Fallback: prompt length + generated length
                    self.cached_len = full_ids.shape[1] + (out.sequences.shape[1] - full_ids.shape[1])

            return "".join(collected)
        else:
            out = self.model.generate(**gen_kwargs)
            if hasattr(out, "past_key_values") and out.past_key_values is not None:
                self.cache = out.past_key_values
                try:
                    self.cached_len = self.cache.get_seq_length()
                except Exception:
                    self.cached_len = full_ids.shape[1] + (out.sequences.shape[1] - full_ids.shape[1])

            # sequences include the single last_tok at the start; slice it off
            gen_ids = out.sequences[0, full_ids.shape[1]:]
            return self.tok.decode(gen_ids, skip_special_tokens=True)

    def ask(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        reply = self._do_generate()
        self.messages.append({"role": "assistant", "content": reply})

        # Optional: re-sync cache to canonical chat_template encoding (robust across turns)
        try:
            enc_full = self._encode_full(add_gen_prompt=False)
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
            pass

        return reply

    def set_system(self, text: str):
        self.messages[0] = {"role": "system", "content": text}

    def clear(self):
        sys_prompt = self.messages[0]
        self.messages = [sys_prompt]
        self.cache = DynamicCache()
        self.cached_len = 0


def main():
    parser = argparse.ArgumentParser(description="Terminal chat with Qwen2.5-3B using generate().")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="HF model id")
    parser.add_argument("--device-map", default="auto", help="transformers device_map")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16"], help="compute dtype")
    parser.add_argument("--attn", default=None, choices=[None, "flash_attention_2"], help="attention backend")
    parser.add_argument("--quant", default=None, choices=[None, "4bit", "8bit"], help="load quantized weights")
    parser.add_argument("--system", default="You are a helpful assistant.", help="system prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="max new tokens per turn")
    parser.add_argument("--temperature", type=float, default=0.7, help="sampling temperature (0 = greedy)")
    parser.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling top_p")
    parser.add_argument("--no-stream", action="store_true", help="disable streaming output")
    args = parser.parse_args()

    engine = GenerateChatEngine(
        model_id=args.model,
        system_prompt=args.system,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_impl=args.attn,
        quant=args.quant,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=not args.no_stream,
    )

    print(
        "\nQwen2.5-3B Terminal Chat (generate)\n"
        f"Model: {args.model}\nSystem: {args.system}\n"
    )
    print("Commands: /help  /clear  /system <text>  /max <n>  /temp <f>  /topp <f>  /quit\n")

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
                    print(
                        "Commands:\n"
                        "  /help               Show this help\n"
                        "  /clear              Clear conversation (keep system prompt)\n"
                        "  /system <text>      Set system prompt\n"
                        "  /max <n>            Set max_new_tokens\n"
                        "  /temp <f>           Set temperature (0.0 = greedy)\n"
                        "  /topp <f>           Set top_p\n"
                        "  /quit               Exit"
                    )
                    continue
                elif cmd == "clear":
                    engine.clear()
                    print("History cleared.")
                    continue
                elif cmd == "system":
                    if arg:
                        engine.set_system(arg)
                        print("System prompt updated.")
                    else:
                        print("Usage: /system <text>")
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

            engine.ask(user)

    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")


if __name__ == "__main__":
    main()
