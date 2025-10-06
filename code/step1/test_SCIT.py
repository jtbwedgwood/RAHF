import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# S3 path and local cache directory
s3_path = "s3://james-cmu-storage/experiments/SCIT"
local_dir = "/workspace/ckpts/hir"

# Sync from S3 to a local directory
os.system(f"aws s3 sync {s3_path} {local_dir} --only-show-errors")

# Load tokenizer and model from the local copy
print(f"Loading model from {local_dir}...")
tokenizer = AutoTokenizer.from_pretrained(local_dir, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

print("Model loaded successfully.")

# Simple prompt
prompt = "Question: What is the capital of France?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

# Decode and print
print("\n=== Model output ===")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
