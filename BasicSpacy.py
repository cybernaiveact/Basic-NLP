import spacy
import random

# Load SpaCy's pre-trained language model
nlp = spacy.load('en_core_web_sm')

# Define the prompt for text generation
prompt = "Hello, how are you doing today?"

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

print(response)
