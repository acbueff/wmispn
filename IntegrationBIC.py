
# coding: utf-8

# In[11]:


import pandas as pd
import os
import numpy as np
from scipy.integrate import quad
from scipy.integrate import dblquad
import scipy.special as special
import math
import pandas as pd
import rpy2.robjects as robjects
get_ipython().magic('reload_ext rpy2.ipython')


# In[12]:


def preprocessUCITXTs(txtfile,namelist=''):
    start=True
    dataAsDict={}
    with open(txtfile,'r') as txt:
        for line in txt:
            #print(line)
            line==line.strip()
            tokens=line.split()
            #build Dictionary
            if start:
                start=False
                if namelist=='':
                    namelist=[]
                    for i in range(len(tokens)):
                        num=i+1
                        name='attr'+str(num)
                        namelist.append(name)
                        dataAsDict[name]=[float(tokens[i].strip())]
                else:
                    for i in range(len(namelist)):
                        dataAsDict[namelist[i]]=[float(tokens[i].strip())]


            else:
                for i in range(len(tokens)):
                    dataAsDict[namelist[i]].append(float(tokens[i].strip()))
    return dataAsDict




# In[15]:


def preprocessCSV(csvfile,separator=' '):
    #print "woooork"
    data_wine={}
    with open(csvfile,'r') as wine:
        start=True
        names=[]
        for line in wine:
            line=line.strip()
            tokens=line.split(separator)
            if start:
                start=False
                for token in tokens:
                    token=token.replace('"','')
                    data_wine[token]=[]
                    names.append(token)
            else:
                for i in range(len(tokens)):
                    data_wine[names[i]].append(float(tokens[i].strip()))
    return data_wine


# In[ ]:





# In[16]:


def genericPolynomial(x,a_06):
    return a_06[0]+a_06[1]*x+a_06[2]*(x**2)+a_06[3]*(x**3)+a_06[4]*(x**4)+a_06[5]*(x**5)+a_06[6]*(x**6)


# In[83]:


"""Note: no directory changes anymore
"""
def calcPolynomials(feature,fileident,name='',nintervals=10,discMethod='equal-width'):
    
    
    print('hello6 calcPoly')
    #print(attrVec)
    BIC_train_scores = []
    best_BIC_score_train = -99999
    #BIC_train = []
    #LL_train = []
    LL_train_scores = []
    attrVec=sorted(feature)
    for poly in range(2,7):

        #attrVec=sorted(feature)
        get_ipython().magic('%R')
        
        get_ipython().magic('%R library("Rmop")')
        get_ipython().magic('%R print("okay")')
        get_ipython().magic('%R vec=attrVec')

        get_ipython().magic('R -i attrVec,poly,discMethod,nintervals n1 <- node.sequence(dim=1,order=poly,data =attrVec, nintervals = nintervals, minval = NULL, maxval = NULL, nodes = NULL, disc.method =discMethod)')
        get_ipython().magic('R -i attrVec -o experiment experiment <- mop.approx(data=attrVec,order=poly,nodeseq=n1,debug=TRUE)')
        #print(experiment)
        get_ipython().magic('R mop <- experiment$mop')
        get_ipython().magic('R -o summary summary<- mop')
        #print(summary)
        get_ipython().magic('R -o cur_polynomial cur_polynomial<-mop$polynomials')

        get_ipython().magic('R -i attrVec -o score score <- mop.score(mop,attrVec)')
        get_ipython().magic('R print(score$BIC)')
        print( "score")
        get_ipython().magic('R -o BIC_train BIC_train<-score$BIC')
        get_ipython().magic('R print(BIC_train)')
        #print BIC_train
        print( "BIC_train")
        BIC_train_scores.append(round(BIC_train[0],4))
        get_ipython().magic('R -o LL_train LL_train<-score$LL')
        LL_train_scores.append(round(LL_train[0],4))
        get_ipython().magic('R -o pieces_train pieces_train<-score$npieces')
        #nr_intervals_array.append(pieces_train[0])
        if BIC_train[0] > best_BIC_score_train:
            best= cur_polynomial
            best_BIC_score_train=BIC_train[0]


        #if LL_train[0] > best_scores['LL_train'][0]:
            #best= cur_polynomial
            #best_scores['LL_train'] = [LL_train[0],disc_method,interval_method,poly]
            #stats.write('better LL train version found')
        if name!='':
            file =name+discMethod+'_in'+str(nintervals)+'_or'+str(poly)
            filename=file
            get_ipython().magic("R -i attrVec,filename mop.plot.experiment(experiment,data=attrVec,filename=filename,format='png')")
        get_ipython().magic('R -i attrVec mop.plot.experiment(experiment,data=attrVec)')

    dataframe = pd.DataFrame({'Nr Polynomials':np.array([2,3,4,5,6]),  'BIC Train': np.array(BIC_train_scores), 'LL Train': np.array(LL_train_scores)})
    dataframe = dataframe[[ 'Nr Polynomials','BIC Train','LL Train']]
    if name != '':

        f_name= fileident + name + '.csv'
        dataframe.to_csv(f_name)


    datafile=fileident+'.data'
    functionfile=fileident+'.functions'
    valuefile=fileident+'.values'
    with open(datafile,'a') as data:
        with open(functionfile,'a')as functions:
            with open(valuefile,'a') as values:
                functions.write('# \n')
                cutpoints=[]
                valueString=''
                start=True
                for interval in best:
                    minVal=interval[1][0]
                    maxVal=interval[2][0]
                    if start:
                        start=False
                        cutpoints.append(minVal)
                        cutpoints.append(maxVal)
                    else:
                        cutpoints.append(maxVal)

                    l=[0,0,0,0,0,0,0]
                    polString=''
                    i=0
                    for pol in interval[0]:
                        l[i]=pol
                        i+=1
                        polString+=str(pol)+' '
                    polString+='\n'
                    functions.write(polString)
                    integral=quad(genericPolynomial,minVal,maxVal,args=(l))
                    val=integral[0]
                    valueString+=str(val)+' '


                valueString+='\n'
                values.write(valueString)

                dataString=''
                for cut in cutpoints:
                    dataString+=str(cut)+' '
                dataString+='\n'
                data.write(dataString)




# In[66]:


import os
import shutil
def runPolynomial(foldername,discDict,intervalNr=10,discMethod='equal-freq'):
    current_directory = os.getcwd()
    appendString=foldername

    final_directory = os.path.join(current_directory, appendString)

    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    else:
        shutil.rmtree(final_directory)           #removes all the subdirectories!
        os.makedirs(final_directory)
    os.chdir(final_directory)
    datafile=foldername+'.data'
    with open(datafile,'w')as data:
        data.write('')
        data.close()
    functionfile=foldername+'.functions'
    with open(functionfile,'w')as functions:
        functions.write('')
        functions.close()
    valuefile=foldername+'.values'
    with open(valuefile,'w')as val:
        val.write('')
        val.close()


    check=foldername+'test'
    #print "discDict: "
    #print discDict
    for attr in discDict:
        calcPolynomials(discDict[attr],foldername,name=attr,nintervals=intervalNr,discMethod=discMethod)

    os.chdir(current_directory)


# In[19]:


def runAllUCI(destination_foldername,txtfile,namelist='',intervalNr=10,discMethod='equal-freq'):
    featureDict=preprocessUCITXTs(txtfile,namelist='')
    runPolynomial(destination_foldername,featureDict,intervalNr=intervalNr,discMethod=discMethod)


# In[24]:


def runAllCSV(destination_foldername,csvfile,separator=' ',intervalNr=10,discMethod='equal-freq'):
    featureDict=preprocessCSV(csvfile,separator)
    runPolynomial(destination_foldername,featureDict,intervalNr=intervalNr,discMethod=discMethod)


# In[78]:

#arguments = (sys.argv)[1:]
#print arguments
#name = arguments[0]
#pathsrc = arguments[1]
#level = arguments[2]
#pathdest = arguments[3]
#onehot = argument[4]

#D:\wmispn\sub-wmispn\data\german
#pathsrc = "/home/andreas/Documents/MScAPR/testzone/australian-dataset.csv"
#D:\wmispn\sub-wmispn\data\german\germanPP
#pathdest = "/home/andreas/Documents/MScAPR/testzone/dataplace"
#os.chdir(r'C:\Users\Steffi\OneDrive - University of Edinburgh\cleaned_MSC')
pathsrc = "D:/wmispn/sub-wmispn/data/german/german-dataset.csv"
pathdest = "D:/wmispn/sub-wmispn/data/german/germanPP"
#os.chdir(r'C:\home\andreas\Documents\MScAPR\SPN-wine\continuousdata')
#os.chdir(r"/home/andreas/Documents/MScAPR/testzone/")
os.chdir(r"D:/wmispn/sub-wmispn/data/german/")
# In[84]:


runAllCSV(pathdest,'german-dataset.csv',separator=',', discMethod='equal-width')



# In[ ]:
