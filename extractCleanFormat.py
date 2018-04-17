import os
import zipfile
import sqlite3
import re
# import unicodedata


# Unzip DB
if os.path.exists('./data/database.sqlite.zip'):
    with zipfile.ZipFile('./Data/database.sqlite.zip') as myzip:
        myzip.extractall('./Data')
        del myzip
else:
    raise Exception(
        'Download Pitchfork Reviews DB from Kaggle')

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

def normalize_unicode(query):
    data = []
    for row in query:
        # Get rid of formatting and compact older reviews
        data.append(re.sub('\s+', ' ', row[0]))
        # data.append(unicodedata.normalize("NFKD", row[0].strip()))
    return data

pitchfork = normalize_unicode(pitchfork)


# Create sentence tokes using regex, store sentence tokens in list
# List can be used for processing but we'll write out a text file
# Text file can be streamed to interator to feed gensim word2vec model
# Reading back in will also make sure we're fully unicode regularized

# Function to split sentences using regex and create list of strings
# where every string is a sentence in a review
def sentence_tokenize(corpus):
    sentences = []
    for i in range(len(corpus)):
        temp = re.split('(?:(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?)\s|(?<=[.!?][\"”]) +)', corpus[i])
        for k in range(len(temp)):
            sentences.append(temp[k])
    sentences = list(filter(None, sentences))
    return sentences

pitchfork_sentences = sentence_tokenize(pitchfork)

# There are 488 sentences with escapes followed by non-word characters
# Hand sampling / search shows they are all embedded in the DB and site text
# 16 sentences have \t, javascript errors in old reviews,
# all quoting outside text. Those are of no consequence
# There are 476 sentences with an \x* replacing words with
# non-english characters not coded in unicode.
# Sadly, those sentences with have to be removed to retain accuracy.

regex = re.compile(r'\\[a-z]')
errors = [i for i in pitchfork_sentences if regex.search(i)]
len(errors)

pitchfork_sentences = [i for i in pitchfork_sentences if not regex.search(i)]

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
