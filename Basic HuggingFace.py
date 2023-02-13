from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Prompt the user to enter a text prompt
prompt = input("Enter a prompt: ")

# Generate text based on the prompt
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids, max_length=1000, do_sample=True)

# Print the generated text
print(tokenizer.decode(output[0], skip_special_tokens=True))
