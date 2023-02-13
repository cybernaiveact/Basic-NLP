import torch
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.training.trainer import Trainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.models import Model

# Define the model class
class LanguageModel(Model):
    def __init__(self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        self.hidden2output = torch.nn.Linear(in_features=encoder.get_output_dim(), out_features=vocab.get_vocab_size('tokens'))
    
    def forward(self, tokens):
        embedded = self.embedder(tokens)
        encoded = self.encoder(embedded)
        output = self.hidden2output(encoded)
        return {'logits': output}
    
    def get_metrics(self, reset: bool = False):
        return {}

# Define the dataset reader
reader = TextClassificationJsonReader(tokenizer=WordTokenizer(),
                                       token_indexers={'tokens': SingleIdTokenIndexer(namespace='tokens')})

# Load the dataset
train_dataset = reader.read('train.jsonl')
dev_dataset = reader.read('dev.jsonl')
test_dataset = reader.read('test.jsonl')

# Create the vocabulary
vocab = Vocabulary.from_instances(train_dataset + dev_dataset + test_dataset)

# Create the embedding layer
embedding_dim = 100
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=embedding_dim)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

# Create the encoder
encoder = BagOfEmbeddingsEncoder(embedding_dim=embedding_dim)

# Create the model
model = LanguageModel(vocab=vocab, embedder=word_embeddings, encoder=encoder)

# Create the iterator
iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])

# Define the optimizer
optimizer = AdamOptimizer(model_parameters=model.parameters())

# Create the trainer
trainer = Trainer(model=model, optimizer=optimizer, iterator=iterator, train_dataset=train_dataset,
                  validation_dataset=dev_dataset, patience=10, num_epochs=100)

# Train the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate(test_dataset)
print(metrics)
