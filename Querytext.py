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
#from gui import *
with open('InvertedIndexing.txt') as f:
    total_indx = json.load(f)


def one_word_query(word, invertedIndex):
    pattern = re.compile('[\W_]+')
    word = pattern.sub(' ', word)
    if word in invertedIndex.keys():
        return [int(filename) for filename in invertedIndex[word].keys()]
    else:
        return []


def Intersection(lst1, lst2):
    return set(lst1).intersection(lst2)
def free_text_query(string):
    pattern = re.compile('[\W_]+')
    string = pattern.sub(' ', string)
    result = []
    kun=[]
    pen=[]
    c=0
    for word in string.split():
        pen= one_word_query(word, total_indx)
        result+=pen
        if(c==0):
            kun=pen
        else:
            kun=Intersection(kun,pen)
        c+=1
    #print(kun)
    print(len(kun))
    return list(set(result))


def phrase_query(string):
    pattern = re.compile('[\W_]+')
    string = pattern.sub(' ', string)
    listOfLists, result = [], []
    for word in string.split():
        listOfLists.append(one_word_query(word,total_indx))
    setted = set(listOfLists[0]).intersection(*listOfLists)
    for filename in setted:
        temp = []
        for word in string.split():
            temp.append(total_indx[word][str(filename)][:])
        for i in range(len(temp)):
            for ind in range(len(temp[i])):
                temp[i][ind] -= i
        if set(temp[0]).intersection(*temp):
            result.append(filename)
    return result


'''print(str(datetime.datetime.now()))
query = input()
result = []
result = one_word_query(query, total_indx)
#for index in result:
#   print(data[index])
# print(data[result[0]])
print(len(result))
query2 = input("enter the text\n")
result = free_text_query(query2)
print(result)
print(query2)
pattern = re.compile('[\W_]+')
query2 = pattern.sub(' ', query2)
query2 = query2.split()
print(query2)
fi = filepreprocessing(query2)
queryl = fi.getdata()
print(queryl)
fin = {}
tfi = tfidfcal()
fin = tfi.getranking(queryl, result)
final = [i[0] for i in fin]
print(len(final))'''
# print(data[result[0]])


# doc1=["this is a mango,Mango is the known as king of friuts,It has is mostly yellow in color,mujhe accha lagta hai bachpan me bahut khata tha","Apple is mostly found in Himanchal pradesh in india"]
# doc1=["hi apple","hello ball"]

# search_input = argv
# code in question






'''def callLists():
    string=getInputQuery()
    result = free_text_query(string)
    pattern = re.compile('[\W_]+')
    string = pattern.sub(' ', string)
    string = string.split()
    fi = filepreprocessing(string)
    queryl = fi.getdata()
    fin = {}
    tfi = tfidfcal()
    fin = tfi.getranking(queryl, result)
    final = [i[0] for i in fin]
    print(len(final))
    return final'''



