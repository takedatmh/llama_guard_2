from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-Guard-2-8B"
device = "cpu" # or "cuda" for GPU
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

def moderate(chat):
    print("1")
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    print("2") 
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    print("3")
    prompt_len = input_ids.shape[-1]
    print("4")
    print(tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True))
    print("5")
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

print("6")
moderate([
    {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
    {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},
])
print("7")
# `safe`
