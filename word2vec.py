import gensim

sentences = word2vec.Text8Corpus((root_path + filename).strip('.zip'))

model = Word2Vec(sentences, size=300, window=5, min_count=5, workers=8)