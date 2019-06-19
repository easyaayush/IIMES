from tfidf import *
from filepreprocessing import *
from finalData import *
class feedback_score:
    def __init__(self):
        self.name="kundan"

    def getresult(self):
        tfres=tfidfcal()
        #res={}
        #res=tfres.getscore_matrix()
        final=[res1[1] for res1 in self.res]
        print(final)
        feedback=[]

    def recieve_feedback(self,feedbackfiles,fin):
        length=len(fin)
        bar=dict(fin)

        for i in range(len(feedbackfiles)):
            bar[feedbackfiles[i]]=bar[feedbackfiles[i]]+0.7
        if(len(fin)>30):
            for i in range(21,30 ):
                bar[fin[i][0]]=bar[fin[i][0]]+0.6
        sorted_by_value = sorted(bar.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_by_value
        '''t={}
        i=0
        for doc in bar.keys():
            t[doc]=0.5-i*(0.5/length)
            i=i+1
        for i in range(len(feedbackfiles)):
            bar[feedbackfiles[i]]=1/2*t[feedbackfiles[i]]-1/2*0.5
        for j'''

#feedbackfiles=[3947,4226]



