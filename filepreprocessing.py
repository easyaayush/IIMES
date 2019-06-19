import re
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
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 'may', 'can', 'not', 'one'])
class filepreprocessing:
    def __init__(self,data):
        data = [re.sub('\s+', ' ', sent) for sent in data]
        data = [re.sub("\'", "", sent) for sent in data]
        data_words = list(self.tokenize_words(data))
        # print(data_words[:1])
        #bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
        #trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

        # Faster way to get a sentence clubbed as a trigram/bigram
        #self.bigram_mod = gensim.models.phrases.Phraser(bigram)
        #self.trigram_mod = gensim.models.phrases.Phraser(trigram)

        # See trigram example
        # print(trigram_mod[bigram_mod[data_words[0]]])
        data_words_nostops = self.remove_stopwords(data_words)
        #data_words_bigrams = self.make_bigrams(data_words_nostops)
        #nlp = spacy.load('en', disable=['parser', 'ner'])
        #self.data_lemmatized = self.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        #print(self.data_lemmatized)
        self.data_lemmatized=data_words_nostops

    def tokenize_words(self,sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))

    def remove_stopwords(self,texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(self,texts):
        return [self.bigram_mod[doc] for doc in texts]

    def make_trigrams(self,texts):
        return [self.trigram_mod[self.bigram_mod[doc]] for doc in texts]

    def lemmatization(self,texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    def getdata(self):
        return self.data_lemmatized

