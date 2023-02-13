from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn as nn

# Define the device to use for training (e.g. CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Parallelize the model across multiple GPUs
model = nn.DataParallel(model)

# Move the model to the device
model.to(device)

# Prompt the user to enter a text prompt
prompt = input("Enter a prompt: ")

# Encode the prompt as input IDs and move to the device
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

# Generate text based on the prompt
with torch.no_grad():
    output = model.generate(input_ids, max_length=1000, do_sample=True)

# Decode the generated output and print it
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
