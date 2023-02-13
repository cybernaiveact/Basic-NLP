import spacy
import random
from transformers import pipeline

# Load SpaCy's pre-trained language model
nlp = spacy.load('en_core_web_sm')

# Load the GPT-2 language model from Hugging Face
generator = pipeline('text-generation', model='gpt2')

# Define the prompt for text generation
prompt = "I love spending time in the great outdoors."

# Generate the response text
response = ""
doc = nlp(prompt)
for token in doc:
    if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]:
        similar_words = [word.text for word in token.similar()]
        if similar_words:
            response += random.choice(similar_words) + " "
        else:
            response += token.text + " "
    else:
        response += token.text + " "

# Generate additional text using the GPT-2 model
generated_text = generator(response, max_length=50, num_return_sequences=1, temperature=0.8)[0]['generated_text']

# Print the final response
print(generated_text)
