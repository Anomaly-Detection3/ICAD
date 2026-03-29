import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

import nltk
nltk.data.path.append('../word2vec/nltk_data')

text_all = [
    "Natural language processing is powerful.",
    "Word embeddings capture semantic meanings.",
    "Gensim makes training easy."
]


tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in text_all]

# tokenized_sentences = [sentence.lower().replace('.', '').replace(',', '').split() for sentence in text_all]

model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=256,
    window=5,
    min_count=1,
    sg=0,
    epochs=100
)


sentence_vectors = []
for sentence in tokenized_sentences:
    vectors = []
    for word in sentence:
        if word in model.wv:
            vectors.append(model.wv[word])
    if vectors:
        sentence_vector = np.mean(vectors, axis=0)
    else:
        sentence_vector = np.zeros(model.vector_size)
    sentence_vectors.append(sentence_vector)


print(len(sentence_vectors))
print(sentence_vectors[0][:5])


