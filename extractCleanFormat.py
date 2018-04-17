import  os
import zipfile
import sqlite3
import re
#import unicodedata


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
        #data.append(unicodedata.normalize("NFKD", row[0].strip()))
    return data

pitchfork = normalize_unicode(pitchfork)


# Write out compacted reviews as txt file
# with line break separating each review
with open('./Data/pitchfork.txt', 'w') as file:
    for review in pitchfork:
        file.write("%s\n" % review)
    del(file, review)


# Create sentence tokes using regex, store sentence tokens in list
# List can be used for processing but we'll write out a text file
# Text file can be streamed to interator to feed gensim word2vec model
# Reading back in will also make sure we're fully unicode regularized

# Function to split sentences using regex and create list of strings
# where every string is a sentence in a review
def sentence_tokenize(corpus):
    sentences = []
    for i in range(len(corpus)):
        temp = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?)\s|(?<=[.!?][\"”]) +', corpus[i])
        for k in range(len(temp)):
            sentences.append(temp[k])
    return sentences

pitchfork_sentences = sentence_tokenize(pitchfork)


# Write out single sentences as txt file
# with line break separating each review
with open('./Data/pitchfork_sentences.txt', 'w') as file:
    for sen in pitchfork_sentences:
        file.write("%s\n" % sen)
    del(file, sen)


# Delete variables if called as script
del(pitchfork, pitchfork_sentences)





