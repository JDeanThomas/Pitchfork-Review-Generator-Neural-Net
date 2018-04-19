import gensim

from gensim.models import word2vec
import logging
import numpy as np


def index_vocab(corpus, wv):
    index = []
    for sentence in corpus:
        for word in sentence:
            if word in wv:
                index.append(wv.vocab[word].index)
    return index

def make_embedding_matrix(model, vector_dim):
    # convert the wv word vectors into a numpy matrix
    embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def preprocess(filename):
    # Longest non-coined, non-technical word in the Oxford English Dictionary
    bound = len('Antidisestablishmentarianism')
    corpus = []
    with open(filename, "r") as f:
        for sen in f.readlines():
            #corpus.append(gensim.utils.simple_preprocess(sen, min_len=1, max_len=bound))
            corpus.append(simple_preprocess(sen, min_len=1, max_len=bound))
        return corpus

pitchfork_sentences = preprocess('./Data/pitchfork_sentences.txt')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = word2vec.Word2Vec(pitchfork_sentences, size=300, window=5, min_count=2, iter=10,
                                     sample=0.00001, sg=1, negative=20,
                                     compute_loss=True, workers=8)

model.save("pitch2vec")

vocab_size = len(model.wv.vocab)
print(model.wv.index2word[0], model.wv.index2word[1], model.wv.index2word[2])
print(model.wv.index2word[vocab_size - 1], model.wv.index2word[vocab_size - 2],
      model.wv.index2word[vocab_size - 3])
print('Index of "of" is: {}'.format(model.wv.vocab['of'].index))
# Similarity
print(model.wv.similarity('woman', 'man'), model.wv.similarity('man', 'king'))
# What doesn't fit?
print(model.wv.doesnt_match("green blue red zebra".split()))
# Most similar words in model vocab
print(model.wv.most_similar(positive="rock"))

# Create list of integer indexes aligning with the model indexes
vocab_index = index_vocab(pitchfork_sentences, model.wv)

print(str_data[:4], index_data[:4])

# Convert the wv word vectors into a numpy matrix
embeddings = make_embedding_matrix(model.wv, 300)