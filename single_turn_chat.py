import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "给我一段关于KV缓存的简短介绍。"}
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # Qwen 官方用法
inputs = tok([text], return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=200)
print(tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
