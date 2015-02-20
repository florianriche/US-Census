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

plt.style.use('ggplot')
'''Loading datas'''
columns_labels= [lines.rstrip() for lines in open("data/columns_name.txt","r")]
new_na_values=['NA','Do not know'," Not in universe"," ?"]
data = pd.read_csv("data/census_income_learn.csv",sep=",",header=0,names=columns_labels,na_values=new_na_values)
data = data.replace(" 50000+.",1)
data = data.replace(" - 50000.",0)

'''Setting of main variables'''
'''PLOT OF SEVERAL COLUMNS'''
#AGE
age = data[['AAGE','INCOME']]

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
#data=data.astype('category')
Y = data.INCOME
X = data[:-1]
##REALISATION DU MODELE LOGISTIQUE
#
#logModel = linear_model.LogisticRegression()
#logModel.fit(X,Y)

#REALISATION DU MODELE FOREST 
forest = RandomForestClassifier(n_estimators=1000,max_depth=20)
forest.fit(X,Y)