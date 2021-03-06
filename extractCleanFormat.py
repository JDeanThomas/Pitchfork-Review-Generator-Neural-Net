import os
import zipfile
import sqlite3
import re
import unicodedata
# Sentence tokenizer from local tokenizer module
from tokenizers import sentence_tokenizer


# Unzip DB
if os.path.exists('./data/database.sqlite.zip'):
    with zipfile.ZipFile('./Data/database.sqlite.zip') as myzip:
        myzip.extractall('./Data')
        del myzip
else:
    raise Exception('Download Pitchfork Reviews DB from Kaggle')

# Connect to DB, query and extract text reviews
conn = sqlite3.connect("./Data/database.sqlite")
conn.text_factory = str
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
cur.execute("PRAGMA table_info(content)").fetchall()

cur.execute("SELECT content FROM content")
pitchfork = cur.fetchall()
print(pitchfork[0])
conn.close()
del(conn, cur)


# Normalize funky formatting of old reviews

def normalize_corpus(query):
    data = []
    for row in query:
        unicodedata.normalize("NFKD", row[0].strip())
        # Get rid of formatting and compact older reviews
        data.append(re.sub('\s+', ' ', row[0]))
    return data

pitchfork = normalize_corpus(pitchfork)


# Create sentence tokens using sentence_tokenizer from local tokenizer
# module. Stores sentence tokens in list of list of reviews
# Sentences object can be used for processing but we'll write out a text file
# Text file can be streamed to interator to feed gensim word2vec model
# Reading back in will also make sure we're fully unicode regularized

pitchfork_sentences = sentence_tokenizer(pitchfork)


# There are 503 sentences with escapes followed by non-word characters
# Hand sampling / search shows they are all embedded in the DB and site text
regex = re.compile(r'\\[a-z]')
errors = [i for i in pitchfork_sentences if regex.search(i)]
len(errors)

# 10 sentences have \t, javascript errors in old reviews,
# all quoting outside text. Those are of no consequence
regex = re.compile(r'\\t')
errors = [i for i in pitchfork_sentences if regex.search(i)]
len(errors)

pitchfork_sentences = [i for i in pitchfork_sentences if not regex.search(i)]

# There are 490 sentences with a pattern \x[A-Z0-9][A-Z0-9]
# encoding errors of unknown origin
# Inspection shows all can simply be deleted
# without altering any sentence syntax (yay!).
regex = re.compile(r'\\x[A-Z0-9][A-Z0-9]')
errors = [i for i in pitchfork_sentences if regex.search(i)]
len(errors)

pitchfork_sentences = [re.sub(r'\\x[A-Z0-9][A-Z0-9]', '', i) for i in pitchfork_sentences]

# No more encoding errors!
errors = [i for i in pitchfork_sentences if regex.search(i)]
len(errors)
del(regex, errors)


def write_corpus(filename, corpus):
    with open(filename, 'w') as file:
        for element in corpus:
            file.write("%s\n" % element)


# Write out compacted reviews as txt file
# with line break separating each review
write_corpus('./Data/pitchfork_reviews.txt', pitchfork)

# Write out single sentences as txt file
# with line break separating each review
write_corpus('./Data/pitchfork_sentences.txt', pitchfork_sentences)

# Delete variables if called as script
del(pitchfork, pitchfork_sentences)