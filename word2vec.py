import gensim

model = Word2Vec(sentences, size=300, window=5, min_count=5, workers=8)