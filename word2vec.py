import gensim
from gensim.models import word2vec
import logging
import re
import numpy as np


def read_data(filename):
    with open(filename, "r") as f:
        data = f.read().split()
    return data

def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data

#with open("pitchfork.txt") as f:
    #text = f.readlines()
    #text = re.sub(r'(M\w{1,2})\.', r'\1', text)
    #text = re.split(r' *[\.\?!][\'"\)\]]* *', text)


sentences = word2vec.LineSentence('pitchfork.txt')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = word2vec.Word2Vec(sentences, size=300, window=5, min_count=2, iter=10,
                                     sample=0.00001, sg=1, negative=20,
                                     compute_loss=True, workers=8)

model.save("pitch2vec")

vocab_size = len(model.wv.vocab)
print(model.wv.index2word[0], model.wv.index2word[1], model.wv.index2word[2])
print(model.wv.index2word[vocab_size - 1], model.wv.index2word[vocab_size - 2],
      model.wv.index2word[vocab_size - 3])
print('Index of "of" is: {}'.format(model.wv.vocab['of'].index))
# Similarity
print(model.wv.similarity('woman', 'man'), model.wv.similarity('man', 'elephant'))
# what doesn't fit?
print(model.wv.doesnt_match("green blue red zebra".split()))


str_data = read_data("pitchfork.txt")
index_data = convert_data_to_index(str_data, model.wv)

# convert the wv word vectors into a numpy matrix
embedding_matrix = np.zeros((len(model.wv.vocab), 300))
for i in range(len(model.wv.vocab)):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
