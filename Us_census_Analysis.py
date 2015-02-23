# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 00:19:11 2015

@author: florianriche
"""


'''Imports'''
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix

plt.style.use('ggp lot')
'''Loading datas'''
columns_labels= [lines.rstrip() for lines in open("data/columns_name.txt","r")]
new_na_values=['NA','Do not know'," Not in universe"," ?"]
data = pd.read_csv("data/census_income_learn.csv",sep=",",header=0,names=columns_labels,na_values=new_na_values)
	model_features_category = ['ACLSWKR','AHGA','AMARITL','AWKSTAT','ARACE','ASEX','PRCITSHP','SEOTR','VETYN']
model_features_continuous= ['AAGE','AHRSPAY','CAPNET','DIVVAL','NOEMP','WKSWORK']


def preprocessingData(data):
    global model_features_category,model_features_continuous
    data = data.replace(" 50000+.",1)
    data = data.replace(" - 50000.",0)
    data['CAPNET'] = data['CAPGAIN']-data['CAPLOSS']


    return data 

data = preprocessingData(data)
X = data[model_features_category+model_features_continuous]
Y = data.INCOME
le = preprocessing.LabelEncoder()
ss = preprocessing.StandardScaler()
for label in model_features_category:
    X[label] = le.fit_transform(X[label])
for label in model_features_continuous:
    X[label] = ss.fit_transform(X[label])

'''Setting of main variables'''
'''PLOT OF SEVERAL COLUMNS'''
#AGE

age = data[['AAGE','INCOME']]

data.groupby('INCOME').mean().T.plot(kind='bar',stacked=True)
#pd.cross_tabs(row,coluns)
age_Total = age.AAGE.value_counts().sort_index()
age_Sup_50K = age[age.INCOME==1]
age_Below_50K = age[age.INCOME==0]
meanSup_50K = age_Sup_50K.mean()
meanTotal = age.mean()

age_up_proportion = age_Sup_50K.AAGE.value_counts().sort_index()/age_Total

plt.figure()
age_Total.plot( color='r')
age_Sup_50K.AAGE.value_counts().sort_index().plot( color='b')
plt.axvline(meanSup_50K.AAGE , color='b', linestyle='--')
plt.axvline(meanTotal.AAGE, color='r', linestyle='--')
age_up_proportion.plot(secondary_y=True, color='g')


#EDUCATION
education  = data[['AHGA','INCOME']]
education_counts1 = education[education.INCOME==1].AHGA.value_counts()
education_counts2= education[education.INCOME==0].AHGA.value_counts()
education_total = pd.concat([education_counts1,education_counts2],axis=1)
education_total.sort(0,ascending=False).plot(kind='bar',stacked=True, log=True)

#RACE
race = data[['ARACE','INCOME']]
race_counts1 = race[race.INCOME==1].ARACE.value_counts()
race_counts2= race[race.INCOME==0].ARACE.value_counts()
race_total = pd.concat([race_counts1,race_counts2],axis=1)
race_total.sort(0,ascending=False).plot(kind='bar',stacked=True,log=True)
#age.plot(kind='bar',stacked=True)
#age.AAGE.value_counts()

#
#PREPARATION OF THE DATA

#REALISATION DU MODELE FOREST 
#forest = RandomForestClassifier(n_estimators=1000,max_depth=20)
forest = RandomForestClassifier()
forest.fit(X,Y)
forest.score(X,Y)
forest.feature_importances_
Y_pr=forest.predict(X)
confusion_matrix(Y,Y_pr)
#To compare with the score that we had if we 
float(Y.value_counts()[1])/float(Y.count())


X_Prime = X[model_features_continuous]
print X_Prime.head(5)
for label in model_features_category:
    tmp = pd.get_dummies(data[label])
    X_Prime=pd.concat([X_Prime,tmp],axis=1)


logModel = linear_model.LogisticRegression(C=1)
logModel.fit(X_Prime,Y)
Y_predict = logModel.predict(X_Prime)

confusion_matrix(Y,Y_predict)

#TEST DU MODELE
newData =  pd.read_csv("data/census_income_test.csv",sep=",",header=0,names=columns_labels,na_values=new_na_values)
newData,newX,newY = preprocessingData(newData)
forest.score(newX,newY)

newX_Prime = newX[model_features_continuous]
print newX_Prime.head(5)
for label in model_features_category:
    tmp = pd.get_dummies(newData[label])
    newX_Prime=pd.concat([newX_Prime,tmp],axis=1)
logModel.score(newX_Prime,newY)
newY_predict = logModel.predict(newX_Prime)
confusion_matrix(newY,newY_predict)

def getImportantFeatures(features,n_top,n_low):
    dic = {}
    for i in range(len(features)):
        dic[features[i]]=i
    Sortedfeatures = sorted(features,reverse=True)
    top_list=[]
    low_list=[]
    for j in range(n_top):
        top_list.append(dic[Sortedfeatures[j]])
        print top_list
    
    for k in range(n_low):
        low_list.append(dic[Sortedfeatures[-k-1]])
        print low_list
        
    return top_list,low_list

'''
