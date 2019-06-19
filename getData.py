from Querytext import *
from finalData import *
from feedback_score import *
from filepreprocessing import *
from finalData import *
def getdata(string):
    #return string
    result = phrase_query(string)
    result += free_text_query(string)
    pattern = re.compile('[\W_]+')
    string = pattern.sub(' ', string)
    string = string.split()
    print("**********************************************")
    print(string)
    fi = filepreprocessing(string)

    queryl = fi.getdata()
    print("##########################################")
    print(queryl)
    datawanted = finalData()
    tfi = tfidfcal()
    fin = tfi.getranking(datawanted,queryl, result)
    '''count = 0
    for des in fin:
        if des[1] > 0.03:
            count += 1
    print("******")
    print(len(fin))
    print(count)
    count = 0
    for des in fin:
        if des[1] > 0.05:
            count += 1
    print("******")
    print(len(fin))
    print(count)'''
    #print(fin)
    finali = fin[0][0]
    #print(finali)
    if(finali=="keyword not present in any file"):
        final=finali
    else:
        final = datawanted.getfinal(fin)


    #final = [i[0] for i in fin]
    # print(len(final))
    #print(final)
    #print("kundan")
    return final
def getexploreddata(string,feedbackfiles):
    result = phrase_query(string)
    result += free_text_query(string)
    pattern = re.compile('[\W_]+')
    string = pattern.sub(' ', string)
    string = string.split()
    fi = filepreprocessing(string)
    queryl = fi.getdata()
    tfi = tfidfcal()
    fin = tfi.getranking(queryl, result)
    feed=feedback_score()
    fin=feed.recieve_feedback(feedbackfiles,fin)

    '''count = 0
    for des in fin:
        if des[1] > 0.03:
            count += 1
    print("******")
    print(len(fin))
    print(count)
    count = 0
    for des in fin:
        if des[1] > 0.05:
            count += 1
    print("******")
    print(len(fin))
    print(count)'''
    # print(fin)
    finali = fin[0][0]
    print(finali)
    if (finali == "keyword not present in any file"):
        final = finali
    else:
        f = finalData()
        final = f.getfinal(fin)

    # final = [i[0] for i in fin]
    # print(len(final))
    # print(final)
    # print("kundan")
    return final
getdata("computer vision")
