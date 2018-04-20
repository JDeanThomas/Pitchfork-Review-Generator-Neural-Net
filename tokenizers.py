import re
import sys

#if sys.version_info[0] >= 3:
    #unicode = str



def sentence_tokenizer(corpus):
    """Sentence tokenizer splits sentences using regex and create list of strings
    where every string is a sentence from a corpus document"""

    sentences = []
    for i in range(len(corpus)):
        #temp = re.split('(?:(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?)\s|(?<=[.!?][\"”]) +)', corpus[i])
        temp = sent_tokenize(corpus[i])
        for k in range(len(temp)):
            sentences.append(temp[k])
    sentences = list(filter(None, sentences))
    return sentences


def word_tokenizer(text, lowercase=False, encoding='utf8', errors="strict"):
    """Word tokenizer that uses regex to tokenize words from unicode sentence strings.
    Iteratively yields contiguious sequence of tokens from sentence input of type str

    Regext was hand coded for this model. Captures whole semantic sequences of words not
    captured by other tokenizers. Defaults to retaining capitalization and captures words
    with punctuation, like possessives, single-hyphenated words and number words like 3rd"""

    TOKENS = re.compile(r'[[a-zA-Z]+[\'\’][a-zA-Z]+|[a-zA-Z]+[\-][a-zA-Z]+|[\d\'\’]+[a-zA-Z]+|[a-zA-Z]+',
                        re.UNICODE)
    #if isinstance(text, unicode):
        #text = unicode(text, encoding, errors=errors)
    lowercase = lowercase
    if lowercase:
        text = text.lower()
    for match in TOKENS.finditer(text):
        yield match.group()


def tokenize_words(corpus, lowercase=False, min_len=1, max_len=30, errors='strict'):
    """Convert a string or list of stings representing sentences into a list of tokens
    Calls word_tokenizer. Can be called on strings, list of strings or corpus via function"""

    tokens = [token for token in word_tokenizer(corpus, lowercase=lowercase, errors=errors)
              if min_len <= len(token) <= max_len and not token.startswith('_')]
    return tokens


def tokenize_sentence_file(sentence_file):
    """Reads corpus of sentence strings by line from file and produces
    list of lists of word tokenized sentence strings via tokenize_words"""

    # Longest non-coined, non-technical word in the Oxford English Dictionary
    bound = len('Antidisestablishmentarianism')
    corpus = []
    with open(sentence_file, "r") as f:
        for sen in f.readlines():
            corpus.append(tokenize_words(sen, max_len=bound))
    return corpus