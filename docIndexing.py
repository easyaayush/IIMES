from dataabs import *
from filepreprocessing import *
import sqlite3


class docIndexing:
    def __init__(self):
        # print("ghjfk")
        d = DataAbstract()
        self.df = d.recieve_dataFrame()
        self.data = self.df.Abstract.values.tolist()
        self.data = self.data[:1]
        p = filepreprocessing(self.data)
        data_lemmatized = p.getdata()
        self.termlistss = {}
        for idx, item in enumerate(data_lemmatized):
            self.termlistss[idx] = item
        self.Indexing()

    def Indexing(self):
        self.conn = sqlite3.connect('InverIndex.db')
        # self.conn.execute('''create table regdex
        #                    (filename Text NOT NULL,
        #                   worddict Text NOT NULL);''')
        print("created table succesfully")
        self.makeindex()

    def makeindex(self):
        total = {}
        for file in self.termlistss.keys():
            total[file] = self.index_one_file(self.termlistss[file])
            # print("before")
            cursor = self.conn.cursor()
            row_count = cursor.execute("select * from regdex where filename=" + "\"" + str(file) + "\";")
            # print("high")

            '''if row_count <= 0:
                print("Not")
            else:
                print("yes")'''

    def index_one_file(self, termlist):
        fileIndex = {}
        for index, word in enumerate(termlist):
            if word in fileIndex.keys():
                fileIndex[word].append(index)
            else:
                fileIndex[word] = [index]
        return fileIndex


g = docIndexing()
