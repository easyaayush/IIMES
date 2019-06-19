import pandas as pd
class DataAbstract:
    __filename='Data.xlsx'
    def __init__(self):
        xl=pd.ExcelFile(self.__filename)
        self.df=xl.parse('abstracts')
        del self.df['Y1']
        del self.df['Y2']
        del self.df['Y']
        del self.df['Domain']
        del self.df['area']
        del self.df['keywords']

    def loadData(self):
        print(self.df.head())
        print(self.df.describe())
    def recieve_dataFrame(self):
        return self.df

'''D1=DataAbstract()
data = D1.df.Abstract.values.tolist()
data = data[:10]
target=open('test.txt','a')
for d in data:
    l=d.split()
    dict={}
    for datas in l:
        if datas not in dict:
            dict[datas]=1
        else:
            dict[datas]=dict[datas]+1
    total=len(l)
    print(total)
    target.write(str(total))
    target.write("\n\n\n")
    for key in dict.keys():
        print(key,"--->",dict[key])
        target.write(str(key))
        target.write("----->")
        target.write(str(dict[key]))
        target.write("\n")
    #target.write(str(dict))
    target.write("\n\n\n")
    target.write("***************************************")

    print("****************************************************************")'''
#D1.loadData()
