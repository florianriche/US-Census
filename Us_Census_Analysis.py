# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# US CENSUS ANALYSIS

# <headingcell level=2>

# OBJECTIVES

# <markdowncell>

# The goal of this project is to describe the expected income of US inhabitants based on their caracteristics such as education, age, gender...
# 
# The goal field is the *total person income* which is defined by the US bureau of economic analysis as : 
# >*Income received by persons from all sources. It includes income received from participation in production as well as from government and business transfer payments. It is the sum of compensation of employees (received), supplements to wages and salaries, proprietors' income with inventory valuation adjustment (IVA) and capital consumption adjustment (CCAdj), rental income of persons with CCAdj, personal income receipts on assets, and personal current transfer receipts, less contributions for government social insurance.[1]*
# 
# Hence, we need not only to take into account the work revenus but also the capital revenues as well as potential social benefits.

# <headingcell level=2>

# DATASET

# <markdowncell>

# The Dataset is composed of 42 Variables and more than 300 000 observations:
# 
# * The learning set is composed of 199523 observations
# 
# * The test set is composed of 99762 observations
# 
# |Description|Code|Example|
# |-----------------------------|-----|
# | Age						|AAGE|25|
# | Class of worker				|ACLSWKR|Self-employed|
# | Industry code					|ADTIND|4|
# | Occupation code				|ADTOCC|34|
# | Education					|AHGA| Some college but no degree |
# | Wage per hour					|AHRSPAY|870|
# | Enrolled in edu inst last wk			|AHSCOL|High School
# | Marital status				|AMARITL|Divorced|
# | Major industry code				|AMJIND|Construction|
# | Major occupation code				|AMJOCC|Sales|
# | Race						|ARACE|White|
# | Hispanic Origin				|AREORGN|Mexican-American|
# | Sex						|ASEX|Male|
# | Member of a labor union			|AUNMEM|No|
# | Reason for unemployment			|AUNTYPE|Job loser - on layoff|
# | Full or part time employment stat		|AWKSTAT|Full-time schedules|
# | Capital gains					|CAPGAIN|1500|
# | Capital losses				|CAPLOSS|200|
# | Divdends from stocks				|DIVVAL|170|
# | Tax filer status				|FILESTAT|NonFiler|
# | Region of previous residence			|GRINREG|Midwest|
# | State of previous residence			|GRINST|Kentucky|
# | Detailed household and family stat		|HHDFMX|HouseHolder|
# | Detailed household summary in household	|HHDREL|child under 18 never married|
# | Instance weight				|MARSUPWT|151.394|
# | Migration code-change in msa			|MIGMTR1|NonMover|
# | Migration code-change in reg			|MIGMTR3|Same Country|
# | Migration code-move within reg		|MIGMTR4|Different country same state|
# | Live in this house 1 year ago			|MIGSAME|Yes|
# | Migration prev res in sunbelt			|MIGSUN|No|
# | Num persons worked for employer		|NOEMP|4|
# | Family members under 18			|PARENT|Mother only present|
# | Country of birth father			|PEFNTVTY|Mexico|
# | Country of birth mother			|PEMNTVTY|United-States|
# | Country of birth self				|PENATVTY|United-States|
# | Citizenship					|PRCITSHP| Native-Born in the United-States|
# | Own business or self employed			|SEOTR|1|
# | Fill inc questionnaire for veteran's admin	|VETQVA|No|
# | Veterans benefits				|VETYN|1|
# | Weeks worked in year				|WKSWORK|48|
# | Year of census |CYEAR|94|
# |Total person Income (target)|PTOTVAL|+ 50000.|

# <headingcell level=2>

# PREPARATION

# <markdowncell>

# Import the required packages

# <codecell>

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import decomposition
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

%matplotlib inline

# <markdowncell>

# We use the codes to name the columns when we load the file and we set the values that are not in the Universe to NaN. Depending on the columns, the NaN labels are differents.

# <codecell>

columns_labels= [lines.rstrip() for lines in open("data/columns_name.txt","r")]
new_na_values=['NA','Do not know',' Not in universe'," Not in universe under 1 year old"," Not in universe or children", " Children or Armed Forces", " ?"]
data = pd.read_csv("data/census_income_learn.csv",sep=",",index_col=False,header=0,names=columns_labels,na_values=new_na_values)

# <markdowncell>

# We must now replace some values to make them more explicit.

# <codecell>

#0-1-2 Multiple choice mapping
dic_MCQ = {0:pd.np.NaN,1:"Yes",2:"No"}
#Company sizes mapping
dic_CompanyPop = {0:pd.np.NaN,1:"Under 10",2:"10 - 24",3:"25 - 99",4:"100-499",5:"500-999",6:"1000+"}

#Based on http://www.census.gov/prod/techdoc/cps/cpsmar96.html
data = data.replace({"VETYN":dic_MCQ,"SEOTR":dic_MCQ,"NOEMP":dic_CompanyPop})

#Simplification of the Education feature to reduce the number of variables
values_to_replace=[' Children',' 7th and 8th grade',' 10th grade',' 11th grade',' 9th grade',' 5th or 6th grade',' 12th grade no diploma',' 1st 2nd 3rd or 4th grade',' Less than 1st grade']
data.AHGA = data.AHGA.replace(values_to_replace,'Less than High School')

#Transformation of the target feature to a 0|1 
le = preprocessing.LabelEncoder()
data.PTOTVAL=le.fit_transform(data.PTOTVAL)

# <markdowncell>

# 40 variables is too much; we should remove some.
# 
# * By removing categorical features that have more than, let's say, 10 values:
#     1. Industry and Occupation codes
#     1. Hispanic origin
#     1. Migration code and previous state
#     1. Parents births countries and birth country
#     1. Detailed household information
# 
# * Categories relevant only for the US administration:
#     1. Tax filer status
#     1. Lived in the same house the previous year
#     1. Year of the census
#     
# * Some categories are also redundant:
#     1. Veteran form question and Veteran benefit question. We keep the latter.
#     1. Enroll in education last year is redundant with Education
#     
# We can then distinguish between our categorical features and continuous features

# <codecell>

model_features_continuous = ['AAGE','AHRSPAY','CAPGAIN','CAPLOSS','DIVVAL','WKSWORK']
model_features_category = ['ACLSWKR','AHGA','AMARITL','AWKSTAT','ARACE','NOEMP','AUNMEM','ASEX','HHDREL','PRCITSHP','SEOTR','VETYN']

# <headingcell level=4>

# Continuous features analysis

# <markdowncell>

# Let's analyze our features. For the continuous variables, we see that we have no missing values

# <codecell>

Result = data[model_features_continuous+["PTOTVAL"]]
#We don't have any missing values:
Result.isnull().sum()

# <markdowncell>

# We can get a good summary of the data using *.describe()*

# <codecell>

#Some features stay at 0 for a long time, we display more quantiles
percentiles_to_display= pd.np.arange(0,1.0,0.1)

res = Result.groupby('PTOTVAL').describe(percentiles=percentiles_to_display)
res=res[model_features_continuous]
res

# <markdowncell>

# We are now going to represent that table. We normalize it so that they are on the same scale.

# <codecell>

plt.figure()
#Plot of the <50000 distribution
A=res.loc[0].iloc[1:]
std=A.ix["std"]
mean=A.ix["mean"]
A= ((A-mean)/std).iloc[2:]
A.boxplot()
plt.suptitle("Distribution of different continous variables for the Low Income")
plt.xlabel('Continuous variables')
plt.ylabel('Distribution')
plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
plt.show()
#Plot of the >50000 income distribution
plt.figure()
B=res.loc[1].iloc[1:]
std=B.ix["std"]
mean=B.ix["mean"]
B = ((B-mean)/std).iloc[2:]
B.boxplot()
plt.suptitle("Distribution of different continous variables for the High Income")
plt.xlabel('Continuous variables')
plt.ylabel('Distribution')
plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
plt.show()

# <markdowncell>

# Let's see if we can find some features where there are many differences.

# <codecell>

Means= Result.groupby("PTOTVAL").mean()
Means


Proportion = (Means/Means.sum()).T.sort(columns=1)
Proportion.plot(kind='bar',stacked=True)
plt.suptitle("Proportion of different continous variables")
plt.xlabel('Continuous variables')
plt.ylabel('Proportion')
plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
plt.show()
#.T.plot(kind='bar',stacked=True)
#plt.show()

# <markdowncell>

# We can see that, high income are usally older and that features related to the capital are almost exclusively for the high income.
# 
# Let's see if we can plot the age distribution to confirm the Age Gap.

# <codecell>

age = data[['AAGE','PTOTVAL']]
age_Total = age.AAGE.value_counts().sort_index()
age_Sup_50K = age[age.PTOTVAL==1]
age_Below_50K = age[age.PTOTVAL==0]
meanSup_50K = age_Sup_50K.mean()
meanTotal = age.mean()

age_up_proportion = age_Sup_50K.AAGE.value_counts().sort_index()/age_Total

fig=plt.figure()
age_Total.plot( color='r')
High_Income=age_Sup_50K.AAGE.value_counts().sort_index().plot( color='b',label="High Income")
plt.axvline(meanSup_50K.AAGE , color='b', linestyle='--')
plt.axvline(meanTotal.AAGE, color='r', linestyle='--')
Proportion_Income=age_up_proportion.plot(secondary_y=True, color='g',label = "Proportion of High Income")
fig.suptitle('Age Distribution', fontsize=20)
plt.xlabel('Age')
plt.ylabel('Proportion')
plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)

plt.show()


# <headingcell level=4>

# Categorical features analysis

# <markdowncell>

# This time, it appears that we have many missing values

# <codecell>

Result = data[model_features_category+["PTOTVAL"]]
Result.isnull().sum()

# <codecell>

res=Result.groupby("PTOTVAL")
res.describe()

# <markdowncell>

# Let's see what are the most sparse features.

# <codecell>

Counts_Res= res.count()

NaN_0 = DataFrame(Result[Result.PTOTVAL==0].isnull().sum())
NaN_1 = DataFrame(Result[Result.PTOTVAL==1].isnull().sum())

NaN_0.columns=["NaN_0"]
NaN_1.columns=["NaN_1"]
fig=plt.figure()
Counts_Res= Counts_Res.append(NaN_0.T)
Counts_Res=Counts_Res.append(NaN_1.T)
Counts_Res[model_features_category].T.plot(kind='bar',stacked=True,title="Propotion of NaN values for different features")
fig.suptitle('test titleezaea', fontsize=20)
plt.xlabel('Categorical features')
plt.ylabel('Occurences')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show();

# <markdowncell>

# Due to the very high number of missing values, we are going to remove some features:
# 
# * SEOTR: Self-employment or not
# * AUNMEM: Member of a labor union

# <codecell>

To_reject=['SEOTR','AUNMEM']
for waste in To_reject:
    if waste in model_features_category:
        model_features_category.remove(waste)
model_features_category

# <headingcell level=2>

# TRAINING

# <markdowncell>

# So we know what we keep, we shall now get our explaining Dataframe and our objective feature and normalize them.
# 
# Continuous values must be normalized using a standard Scaler.
# Categorical values must be normalize by encoding them into an Integer.

# <codecell>

le = preprocessing.LabelEncoder()
ss = preprocessing.StandardScaler()
X = data[model_features_category+model_features_continuous]
Y = le.fit_transform(data.PTOTVAL)
for label in model_features_category:
    X[label] = le.fit_transform(X[label])
for label in model_features_continuous:
    X[label] = ss.fit_transform(X[label])

# <markdowncell>

# We are now going to train our dataset using two classification algorithms : *Random Forest classifier* and a *logistic Regression*

# <headingcell level=4>

# Random Forest

# <codecell>

forest = RandomForestClassifier(n_estimators=30)
forest.fit(X,Y)
Y_predict = forest.predict(X)
print forest.score(X,Y)
print metrics.classification_report(Y,Y_predict)
print metrics.confusion_matrix(Y,Y_predict)
print metrics.roc_auc_score(Y,Y_predict)

# <markdowncell>

# Which is pretty good.

# <headingcell level=4>

# Logistic Regression

# <markdowncell>

# To perform the Logistic Regression, we need to modify again our DataSet by creating dummy_columns for all categorical variables.

# <codecell>

X_log=X[model_features_continuous]
for label in model_features_category:
    dummified_label = pd.get_dummies(data[label],dummy_na=True,prefix_sep=":",prefix=label)
    X_log=pd.concat([X_log,dummified_label],axis=1)
logModel = linear_model.LogisticRegression(C=1)
logModel.fit(X_log,Y)
Y_predict = logModel.predict(X_log)
print logModel.score(X_log,Y)
print metrics.classification_report(Y,Y_predict)
print metrics.confusion_matrix(Y,Y_predict)
print metrics.roc_auc_score(Y,Y_predict)

# <markdowncell>

# Let's do some cross-validation to try to improve our model.

# <codecell>

kf = KFold(len(X_log)-1, n_folds=10,shuffle=True)
Scores=[]
Models=[]
for train,test in kf:
    X_log_train,Y_train,X_log_test,Y_test  = X_log.ix[train], Y[train], X_log.ix[test], Y[test]
    logModel.fit(X_log_train,Y_train)
    Scores.append(metrics.precision_score(logModel.predict(X_log_test),Y_test))
    Models.append(logModel)
#Get the maximum index
BetterScore= Scores.index(max(Scores))
logModel= Models[BetterScore]

# <headingcell level=2>

# MODEL EVALUATION

# <markdowncell>

# Based on what we saw, it seems that the random Forest has a better performance than the logistic regression in terms of prediction. 
# We want to have a good precision and a good recall; the F1-score is a good synthesis and it appears that the Random Forest has a better result.
# 
# Let's see if it works also on the testing set.

# <codecell>

data_test = pd.read_csv("data/census_income_test.csv",sep=",",index_col=False,header=0,names=columns_labels,na_values=new_na_values)
dic_MCQ = {0:pd.np.NaN,1:"Yes",2:"No"}
dic_CompanyPop = {0:pd.np.NaN,1:"Under 10",2:"10 - 24",3:"25 - 99",4:"100-499",5:"500-999",6:"1000+"}

#Based on http://www.census.gov/prod/techdoc/cps/cpsmar96.html
data_test = data_test.replace({"VETYN":dic_MCQ,"SEOTR":dic_MCQ,"NOEMP":dic_CompanyPop})

values_to_replace=[' Children',' 7th and 8th grade',' 10th grade',' 11th grade',' 9th grade',' 5th or 6th grade',' 12th grade no diploma',' 1st 2nd 3rd or 4th grade',' Less than 1st grade']
data_test.AHGA = data_test.AHGA.replace(values_to_replace,'Less than High School')

le = preprocessing.LabelEncoder()
data_test.PTOTVAL=le.fit_transform(data_test.PTOTVAL)
X_test = data_test[model_features_category+model_features_continuous]
Y_test = le.fit_transform(data_test.PTOTVAL)
for label in model_features_category:
    X_test[label] = le.fit_transform(X_test[label])
for label in model_features_continuous:
    X_test[label] = ss.fit_transform(X_test[label])

# <codecell>

Y_test_predicted= forest.predict(X_test)
print metrics.classification_report(Y_test,Y_test_predicted)
print metrics.roc_auc_score(Y_test,Y_test_predicted)
print metrics.confusion_matrix(Y_test,Y_test_predicted)

# <markdowncell>

# It is not a good as on the learning set.
# We still have a good precision but now we had a very bad recall. 
# When we predict a high income, it generally is one, but we miss a lot of them.

# <headingcell level=2>

# INSIGHTS

# <markdowncell>

# We tested our model; it is now time to interpret it. Based on the features importances we can know which values are the most important.

# <codecell>

Results = pd.DataFrame(zip(X.columns, forest.feature_importances_.T))
Results.sort(columns=1,ascending=False)

# <markdowncell>

# We can see that Age, capital gains and dividends as well as education are the primarly explaining factors.
# 
# The fact that age is important is consistent with our earlier chart showing that as age advances, people are more likely to earn more than 50k€.
# 
# The importance of capital gains is relative, according to our earlier description of the dataset, at least 80% of the people that have high income don't have any capital gain. However, those wo do are almost guaranteed to have an high income.
# 
# Below are some charts validating our explanation.

# <codecell>

PLOT_NOEMP= pd.crosstab(data_test.NOEMP,data_test.PTOTVAL)
PLOT_AHGA= pd.crosstab(data_test.AHGA,data_test.PTOTVAL)
PLOT_ACLSWKR= pd.crosstab(data_test.ACLSWKR,data_test.PTOTVAL)
PLOT_ASEX= pd.crosstab(data_test.ASEX,data_test.PTOTVAL)
PLOT_AWKSTAT= pd.crosstab(data.AWKSTAT,data.PTOTVAL)

plt.figure()
(PLOT_NOEMP/PLOT_NOEMP.sum()).sort(columns=1).plot(kind='bar')
plt.suptitle("Income Distribution by Company size")
plt.xlabel('Company Size')
plt.ylabel('Distribution')
plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
plt.show()

plt.figure()
(PLOT_AHGA/PLOT_AHGA.sum()).sort(columns=1).plot(kind='bar')
plt.suptitle("Income Distribution by Education")
plt.xlabel('Education lebel')
plt.ylabel('Distribution')
plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
plt.show()

plt.figure()
(PLOT_ACLSWKR/PLOT_ACLSWKR.sum()).sort(columns=1).plot(kind='bar')
plt.suptitle("Income Distribution by Work Status")
plt.xlabel('Work Status')
plt.ylabel('Distribution')
plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
plt.show()

plt.figure()
(PLOT_ASEX/PLOT_ASEX.sum()).sort(columns=1).plot(kind='bar')
plt.suptitle("Income Distribution by Gender")
plt.xlabel('Gender')
plt.ylabel('Distribution')
plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
plt.show()

plt.figure()
(PLOT_AWKSTAT/PLOT_AWKSTAT.sum()).sort(columns=1).plot(kind='bar')
plt.suptitle("Income Distribution by Employment Status")
plt.xlabel('Employment Status')
plt.ylabel('Distribution')
plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
plt.show()

# <markdowncell>

# From those charts we can deduce that :
#  
#  1. The bigger the company, the more likely a person earns a lot.
#  1. Most of the high incomes have at least a bachelor degree even if it is not compulsory.
#  1. The private sector concentrate the most people (both high and low income). Self-employment seems to give a higher chance of a high income.
#  1. The gender gap appears in our Dataset as men can expect to earn more than women.
#  1. Most of the High income work full-time
# 
# The typical high income is then an educated man, probably working full-time in a big company in private sector or by himself.

# <headingcell level=2>

# Conclusion

# <markdowncell>

# We got a model that overfitted the learning set and is not wrong when it dares to predicts but lacks some features to find all targets.
# 
# There is probably more work to do on the Data cleaning. They are too much children that appear in the model and some features have maybe be discareded too quickly in the beginning of the process.
# 
# The na_values should maybe be handled differently, if I had to continue this work further, I would probably try to see if I can fill the missing data rather than just set them to NaN.
# 
# The challenging tasks were clearly to clean the dataset as well as finding a way to visually explore the data.

