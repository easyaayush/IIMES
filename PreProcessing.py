import re
import dataabs
from dataabs import *
from filepreprocessing import *
import pandas as pd
import numpy as np
import spacy
from nltk.corpus import stopwords
from pprint import pprint
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim.corpora as corpora
import pyLDAvis
# import pyLDA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import datetime
import pickle
from tfidf import *

# nlp = spacy.load('en')

print(str(datetime.datetime.now()))

'''def tokenize_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 'may', 'can', 'not', 'one'])'''
d = DataAbstract()
df = d.recieve_dataFrame()
# data=df.Abstact.values.tolist()
# data=
# df=pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
# print(df.Abstract.unique())
# print(df.head())
data = df.Abstract.values.tolist()
data = data[:10000]
'''data = [re.sub('\s+', ' ', sent) for sent in data]
data = [re.sub("\'", "", sent) for sent in data]
# pprint(data[:1])
data_words = list(tokenize_words(data))
# print(data_words[:1])
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
# print(trigram_mod[bigram_mod[data_words[0]]])
data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostops)
nlp = spacy.load('en', disable=['parser', 'ner'])
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])'''
p = filepreprocessing(data)
data_lemmatized = p.getdata()
termlists = {}
for idx, item in enumerate(data_lemmatized):
    termlists[idx] = item
#print(data_lemmatized[:1])
# target=open('wordterms.txt','a')
# target.write(str(data_lemmatized))

'''id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])
print(id2word[0])
print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])'''
'''lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]'''


def index_one_file(termlist):
    fileIndex = {}
    for index, word in enumerate(termlist):
        if word in fileIndex.keys():
            fileIndex[word].append(index)
        else:
            fileIndex[word] = [index]
    return fileIndex


def make_indices(termlists):
    total = {}
    for filename in termlists.keys():
        total[filename] = index_one_file(termlists[filename])
    return total


# total_index = {}


def fullIndex(regdex):
    total_index = {}
    for filename in regdex.keys():
        for word in regdex[filename].keys():
            if word in total_index.keys():
                if filename in total_index[word].keys():
                    total_index[word][filename].extend(regdex[filename][word][:])
                else:
                    total_index[word][filename] = regdex[filename][word]
            else:
                total_index[word] = {filename: regdex[filename][word]}
    return total_index


d = data_lemmatized[0]
filIndex = {}
totl = {}
total_indx = {}
totl = make_indices(termlists)
# print(totl)


# totl=make_indices(filIndex)
total_indx = fullIndex(totl)
# target = open('indexing.txt', 'a')
# target.write(str(total_indx))
# file=open('invertedindex','wb')
# pickle.dump(total_indx,file)
#print(total_indx)
#target = open('foronep.txt', 'wb')
#pickle.dump(total_indx,target)
#json=json.dumps(total_indx)
#target=open('for.json','w')
#target.write(json)
#with open('InvertedIndexing.txt','w') as f:
#    json.dump(total_indx,f)
tfi = tfidfcal()
#print(tfi.gettfidf())


