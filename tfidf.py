from filepreprocessing import *
# from PreProcessing import *
from dataabs import *
import math
from finalData import *
import ast
import pickle
import json
import collections
from proximity import *


class tfidfcal:
    def __init__(self):
        self.tf = {}
        self.idf = {}
        self.tfidf = collections.defaultdict()
        #self.ourresult = []
        # with open('foronep.txt','rb') as handle:
        # self.index=pickle.loads(handle.read())
        # s = open('forone.txt', 'r').read()
        # self.index = eval(s)
        # t = open('for.json', 'r')
        # self.index = json.load(t.read())
        with open('InvertedIndexing.txt') as f:
            self.index = json.load(f)
        # print(self.index)
        # self.tfidffile = open('tfidffile.txt', 'a')
        # self.tfmatrix()
        # self.idfmatrix()
        # self.tfcalculator()
        # self.idfcalculator()
        # self.tfidfcalculator()
        # print(self.tf)

    def tfmatrix(self):
        for word in self.index.keys():
            for filename in range(0, 100):
                # filename=str(filename)
                file = str(filename)
                if file in self.index[word].keys():
                    if word in self.tf.keys():
                        self.tf[word][filename] = len(self.index[word][str(filename)])
                    else:
                        self.tf[word] = {}
                        self.tf[word][filename] = len(self.index[word][str(filename)])

                else:
                    if word in self.tf.keys():
                        self.tf[word][filename] = 0
                    else:
                        self.tf[word] = {}
                        self.tf[word][filename] = 0
                    # self.tf[word] = {}
                    # self.tf[word][filename] = len(self.index[word][str(filename)])

    def idfmatrix(self):
        for word in self.index.keys():
            self.idf[word] = len(self.index[word].keys())

    def tfcalculator(self):
        d = DataAbstract()
        df = d.recieve_dataFrame()
        data = df.Abstract.values.tolist()
        data = data[:1000]
        p = filepreprocessing(data)
        data_lemmatized = p.getdata()
        for filename in range(0, 1000):
            l = len(data_lemmatized[filename])
            for word in self.tf.keys():
                for file in self.tf[word].keys():
                    self.tf[word][file] /= l

    def idfcalculator(self):
        for word in self.idf.keys():
            self.idf[word] = math.log(1000 / self.idf[word])

    def tfidfcalculator(self):
        for word in self.tf.keys():
            if word not in self.tfidf.keys():
                self.tfidf[word] = {}
            for file in self.tf[word].keys():
                self.tfidf[word][file] = self.tf[word][file] * self.idf[word]
        self.tfidffile.write(str(self.tfidf))

    def gettfidf(self):
        return self.tfidf

    def getranking(self,datawanted, query, result):
        final = {}
        list = []
        for j in query:
            if len(j) != 0:
                word = j[0]
                list.append(word)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        #print(list)
        #datawanted = finalData()
        for i in result:
            res = 0
            #print(type(i))
            for j in query:
                #print(j)
                if len(j) != 0:
                    word = j[0]
                    tfl = 0
                    idfl = 0
                    if word in self.index.keys():
                        #print(word)
                        if str(i) in self.index[word].keys():
                            tfl = len(self.index[word][str(i)]) / 200
                            #print(tfl)
                    else:
                        tfl = 0
                    if word not in self.index.keys():
                        idfl = 0
                    else:
                        idfl = len(self.index[word].keys())
                    if idfl != 0:
                        res += tfl * (math.log(10000 / idfl))

            final[i] = res
            #print(i,final[i])

            t = 0
            word1 = list[t]
            findist_score = 0
            if word1 not in self.index.keys():
                findist_score = 0
            else:
                file1 = self.index[word1]
                dist = []
                dist.append(0)
                result2 = []
                file1set = set(file1)
                for j in range(t + 1, len(list) - 1):
                    dis = j - t
                    result2 = file1
                    word2 = list[j]
                    if word2 not in self.index.keys():
                        continue
                    else:
                        file2 = self.index[word2]
                        file2set = set(file2)
                        count = 0
                        for key in file1set.intersection(file2set):
                            if key == i:
                                for k in file1set[key]:
                                    for l in file2set[key]:
                                        if l - k == dis:
                                            count += 1
                                dist[dis] = count
            sum = 0
            for p in dist:
                sum += (p / len(dist))
            findist_score = sum
            #print(findist_score)
            final[i] += findist_score

            '''document=datawanted.recieveabstract(i)
            lst=[]
            lst.append(document)
            pre=filepreprocessing(lst)
            print("@#%#$^%$&^%^*&^*&^*")
            doc1=pre.getdata()
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            dt=doc1[0]
            print(final[i])
            maxValue=findingMaxValue(len(list))
            score=minDist(dt,list,maxValue)
            final[i]+=score
            print("kunffhdjsklafjdska")
            print(final[i])
            score=maxDist(dt,list,maxValue)
            final[i]+=score
            print("fgdfsjkamxcnvdmsncmx")
            print(final[i])'''
            #print("*****************************")
            #print(i,final[i])
        if len(final) != 0:
            sorted_by_value = sorted(final.items(), key=lambda kv: kv[1],reverse=True)
            #print(sorted_by_value)
            if len(final) > 100:
                self.ourresult = sorted_by_value[:100]
                return self.ourresult
            else:
                self.ourresult = sorted_by_value
                return self.ourresult
        else:
            return [["keyword not present in any file"]]

    def getscore_matrix(self):
        return self.ourresult
