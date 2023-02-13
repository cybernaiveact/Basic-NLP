import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import brown
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Download the Brown corpus for training the model
nltk.download('brown')

# Load the Brown corpus and create a list of sentences
corpus = brown.sents()

# Use the Tokenizer to tokenize and convert the sentences into sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)

# Pad the sequences to make them all the same length
max_sequence_len = max([len(seq) for seq in sequences])
padded_sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))

# Define the model architecture
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 64, input_length=max_sequence_len))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model on the padded sequences
model.fit(padded_sequences, epochs=20)

# Define the prompt for text generation
prompt = "Amidst the flotsam and jetsam of the recent economic downturn, characterized by mass job losses and economic instability, the notion of a robust economic recovery, with substantial job creation and growth, seems tenuous and elusive, raising the specter of long-term structural unemployment and chronic economic malaise."

# Convert the prompt into a sequence of integers
prompt_sequence = tokenizer.texts_to_sequences([prompt])
padded_prompt_sequence = np.array(pad_sequences(prompt_sequence, maxlen=max_sequence_len, padding='pre'))

# Generate the response text
generated_sequence = model.predict(padded_prompt_sequence)[0]
generated_sequence = np.argmax(generated_sequence)
generated_word = tokenizer.index_word[generated_sequence]

# Print the final response
print(generated_word)
