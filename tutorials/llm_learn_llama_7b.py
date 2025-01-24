# from transformers import pipeline
# import torch

# torch.cuda.empty_cache()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# messages = [
#     {
#         "role" : "user", "content": "who are you?"
#     },
# ]

# pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b", max_new_tokens=20, device=device)
# pipe(messages)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = model.to(device)


prompt = "Once upon a time"

inputs = tokenizer(prompt, return_tensors="pt").to(device)


print("Generating texts...")

outputs = model.generate(
    inputs.input_ids,
    max_length=50,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7,
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated text:")
print(generated_text)