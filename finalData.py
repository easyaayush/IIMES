import pandas as pd
class finalData:
    __filename='Data.xlsx'
    def __init__(self):
        x1=pd.ExcelFile(self.__filename)
        self.df=x1.parse('abstracts')
        del self.df['Y1']
        del self.df['Y2']
        del self.df['Y']
        del self.df['Domain']
        del self.df['area']
        self.finlist=self.df.Abstract.values.tolist()
        print("kundan")
        #del self.df['keywords']
    def recievedataFrame(self):
        return self.df
    def recieveabstract(self,i):
        return self.finlist[i]
    def getfinal(self,fin):
        data=self.df.keywords.values.tolist()
        final=[]
        #print(fin)
        fin=dict(fin)
        for datas in fin:
            lst=[]
            lst.append(datas)
            s_tr=data[datas].split(";")
            lst.append(s_tr[0])
            lst.append(fin[datas])
            lst.append(self.finlist[datas])
            final.append(lst)
        return final
