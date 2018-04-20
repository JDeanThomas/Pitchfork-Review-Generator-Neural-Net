import logging
import numpy as np
from gensim.models import word2vec
# Local tokenizers and corpus file I/O
from tokenizers import word_tokenizer, tokenize_words, tokenize_sentence_file
# Local TensorFlow word2vec validation model
from w2vValidationModel import w2v_tf_validation

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


tokenized_sentences = tokenize_sentence_file('./Data/pitchfork_sentences.txt')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = word2vec.Word2Vec(tokenized_sentences, size=300, window=10, min_count=2, iter=10,
                                     sample=0.00001, sg=1, negative=20,
                                     compute_loss=True, workers=8)

model.save("pitch2vec")


vocab_size = len(model.wv.vocab)

# 20 most common words
print(model.wv.index2word[:19])
# 20 least common words
print(model.wv.index2word[vocab_size - 20:])
# Index of string "music"
print('Index of "music" is: {}'.format(model.wv.vocab['music'].index))
# Similarity
print(model.wv.similarity('woman', 'man'), model.wv.similarity('man', 'king'))
# What doesn't fit?
print(model.wv.doesnt_match("green blue red guitar".split()))
# Most similar words in model vocab
print('Most similar words to "indie" is: ', model.wv.most_similar(positive="indie"))


# Create list of integer indexes aligning with the model indexes
vocab_index = index_vocab(tokenized_sentences, model.wv)

print(tokenized_sentences[0][:4], vocab_index[:4])


# Convert the wv word vectors into a numpy matrix
embeddings = make_embedding_matrix(model.wv, 300)

# TensorFlow model as evaluation model for word2vec model
w2v_tf_validation(embeddings, model.wv)