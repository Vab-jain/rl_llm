# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
# messages = [
#     {
#         "role": "system",
#         "content": "You are a friendly chatbot who always responds in the style of a pirate",
#     },
#     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
# ]
messages = [
    {"role": "system", "content": "You are a bot playing a game of tic-tac-toe, and you play as a cross ('1'). The board is represented as an array of lenght 9. Each grid cell is represented at each index of the array, starting from the top left corner and moving down column-wise."},
    {"role": "system", "content": "The board is represented as folows: [0,0,0,0,0,0,0,0,0]. Here, '0' indicates an empty cell, '1' indicates a cross, and '2' indicates a circle. Please decide your next move and specify the grid cell index where you would place your cross." },
    {"role": "system", "content": "Example state: [0,0,0,0,0,0,0,0,0]. This state represent an empty board."},
    {"role": "system", "content": "Example state: [0,0,0,0,1,0,0,0,0]. This state represent a cross in the middle."},
    {"role": "system", "content": "Example state: [0,0,0,0,1,0,0,0,2]. This state represent a cross in the middle and a circle in the bottom right."},
    {"role": "user", "content": "Current state: [0,0,0,0,0,0,0,0,0]. Give your answer in the format: Current State, what will you play, and next State."},
    # {"role": "assistant", "content": ""},
]


prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
# <|system|>
# You are a friendly chatbot who always responds in the style of a pirate.</s>
# <|user|>
# How many helicopters can a human eat in one sitting?</s>
# <|assistant|>
# ...
