import gensim
from gensim.models import word2vec
import re

def read_data(filename):
    with filename as f:
        data = f.read(f.namelist()[0]).split()
    return data

def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data

#with open("pitchfork.txt") as f:
    #text = f.read()
    #text = re.sub(r'(M\w{1,2})\.', r'\1', text)

#sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)

sentences = word2vec.LineSentence('pitchfork.txt')

model = word2vec.Word2Vec(sentences, size=300, window=5, min_count=2, iter=10, sg=1,
                            negative=True, compute_loss=True, workers=8)

model.save("pitch2vec")

str_data = read_data("pitchfork.txt")
index_data = convert_data_to_index(str_data, model.wv)


# convert the wv word vectors into a numpy matrix
embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
for i in range(len(model.wv.vocab)):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector