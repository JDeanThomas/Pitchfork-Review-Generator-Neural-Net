import tensorflow as tf
import numpy as np


"""Simple TensorFlow based validation model for word2vec model
   Functions solely to:
   1) Validate word2vec model parameters 
   2) Examine model performance:
        A) Validate performance of Pitchfork review data set
        B) Validate with the addition of the last year of scraped reviews added
        C) Validate model with Stanford GloVe embeddings then trained on Pitchfork data 
        
    3) Will further be use to evaluate performance of the more granular Penn Treebank
       tokenizer in a wor2vec model vs costom tokenizer and Facebooks FastTest model """


def w2v_tf_validation(embedding_matrix, wv):
    validation_size = 20  # Random set of words to evaluate similarity on.
    validation_window = 100  # Only pick dev samples in the head of the distribution.
    validation_examples = np.random.choice(validation_window, validation_size, replace=False)
    validation_dataset = tf.constant(validation_examples, dtype=tf.int32)

    # embedding layer weights are frozen to avoid updating
    saved_embeddings = tf.constant(embedding_matrix)
    embedding = tf.Variable(initial_value=saved_embeddings, trainable=False)

    # Cosine similarity
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embeddings = embedding / norm
    validation_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, validation_dataset)
    similarity = tf.matmul(validation_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # call our similarity operation
        sim = similarity.eval()
        # run through each validation example, finding closest words
        for i in range(validation_size):
            validation_word = wv.index2word[validation_examples[i]]
            top_k = 10  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % validation_word
            for k in range(top_k):
                close_word = wv.index2word[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)


tf_model(embeddings, model.wv)