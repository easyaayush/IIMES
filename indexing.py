import re
import dataabs
from dataabs import *
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
nlp = spacy.load('en')

print(str(datetime.datetime.now()))


def tokenize_words(sentences):
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
stop_words.extend(['from', 'subject', 'may', 'can', 'not', 'one'])
d = DataAbstract()
df = d.recieve_dataFrame()
data = df.Abstract.values.tolist()
data = data[:10]
data = [re.sub('\s+', ' ', sent) for sent in data]
data = [re.sub("\'", "", sent) for sent in data]
data_words = list(tokenize_words(data))
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostops)
nlp = spacy.load('en', disable=['parser', 'ner'])
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
d=[]
for index in range(10):
    d.append(data_lemmatized[index])
#d=data_lemmatized[:10]
print(d)
file=open('hypothesis.txt','a')
for datas in d:
    file.write(str(len(datas)))
    file.write("\n")
    dict={}
    for words in datas:
        if words in dict.keys():
            dict[words]=dict[words]+1
        else:
            dict[words]=1
    for key in dict.keys():
        file.write(str(key))
        file.write("--->")
        file.write(str(dict[key]))
        file.write("\n\n")

termlists = {}
#'for idx, item in enumerate(data_lemmatized):
 #'   termlists[idx] = item
#'
#print(data_lemmatized[:1])
#target=open('wordterms.txt','a')
#target.write(str(data_lemmatized))


'''def index_one_file(termlist):
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


filIndex = {}
totl = {}
total_indx = {}
totl = make_indices(termlists)
total_indx = fullIndex(totl)
#target = open('indexing.txt', 'a')
#target.write(str(total_indx))
#file=open('invertedindex','wb')
#pickle.dump(total_indx,file)
print(total_indx)


def one_word_query(word, invertedIndex):
    pattern = re.compile('[\W_]+')
    word = pattern.sub(' ', word)
    if word in invertedIndex.keys():
        return [filename for filename in invertedIndex[word].keys()]
    else:
        return []


def free_text_query(string):
    pattern = re.compile('[\W_]+')
    string = pattern.sub(' ', string)
    result = []
    for word in string.split():
        result += one_word_query(word, total_indx)
    return list(set(result))


def phrase_query(string, invertedIndex):
    pattern = re.compile('[\W_]+')
    string = pattern.sub(' ', string)
    listOfLists, result = [], []
    for word in string.split():
        listOfLists.append(one_word_query(word))
    setted = set(listOfLists[0]).intersection(*listOfLists)
    for filename in setted:
        temp = []
        for word in string.split():
            temp.append(invertedIndex[word][filename][:])
        for i in range(len(temp)):
            for ind in range(len(temp[i])):
                temp[i][ind] -= i
        if set(temp[0]).intersection(*temp):
            result.append(filename)
    return result


print(str(datetime.datetime.now()))
query = input()
result = []
result = one_word_query(query, total_indx)
vectorizer = TfidfVectorizer(stop_words='english')
#vector=[]
res=[]
for index in result:
    res.append(data[index])
vectorizer.fit(res)
print()
#vector = vectorizer.fit_transform(res)
#print(vector.shape)
#print(vectorizer.get_feature_names())'''

'''for index in result:
    print(data[index])
# print(data[result[0]])
query2 = input("enter the text\n")
result = free_text_query(query2)
print(data[result[0]])'''
