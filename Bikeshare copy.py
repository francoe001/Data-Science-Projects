#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 15:54:46 2019

@author: efrancois
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm




from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score,cross_val_predict


#%%

#Importing Hourly File
hbike = pd.read_csv('/Users/efrancois/Desktop/IE MAsters/Term2/AI & ML Statistical Learning & Prediction/Bike-Sharing-Dataset/hour.csv', sep=',')



#%%
#General Information:

hbike.info()
#No null values

hbike = hbike.drop('instant',1)

#%%
#Converting to Date to get general Shape of predictions --> Real Life assumptions (more in the summer, less in the winter)

hbike['dteday'] = pd.to_datetime(hbike.dteday)

hbike[['dteday', 'cnt']].groupby(['dteday']).agg('mean').plot()

#%%

fig, axes = plt.subplots(nrows=3,ncols=1)
fig.set_size_inches(10, 40)

sns.boxplot(data=hbike,y="cnt",x="mnth",orient="v",ax=axes[0])
axes[0].set(xlabel='Month', ylabel='Count',title="Distribution of Users per month")

sns.boxplot(data=hbike,y="cnt",x="hr",orient="v",ax=axes[1])
axes[1].set(xlabel='Hour Of The Day', ylabel='Count',title="Distribution of Users per hour")

sns.boxplot(data=hbike,y="cnt",x="workingday",orient="v",ax=axes[2])
axes[2].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count/Working Day Distribution")

#%%


#Dropping these as we want to predict totals 

hbike = hbike.drop('casual',1)
hbike = hbike.drop('registered',1)



corrMatt = hbike.corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corrMatt.round(2), mask=mask,vmax=.8, square=True,annot=True)


#%%

Xraw = hbike[['yr','hr','season','workingday','atemp','windspeed','holiday','weathersit']]


#One Hot Encoding of categorical values

ohe = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
pipec = Pipeline([ohe])
Xc = hbike[['season','weathersit','hr']]
Xct = pd.DataFrame(pipec.fit_transform(Xc))
Xraw = Xraw.drop('weathersit', 1)
Xraw = Xraw.drop('season', 1)
Xraw = Xraw.drop('hr', 1)

X = pd.concat([Xraw,Xct],axis=1,sort=False)
y = hbike[['cnt']]

X.head()


#%%


#Modelling


cvStrategy = KFold(n_splits=6, shuffle=True, random_state=0)
scoring = 'r2'
model = RandomForestRegressor(n_estimators=500, n_jobs=-1)
#model = linear_model.Ridge()
#model = svm.SVR(kernel='linear', C=1000)
#model = linear_model.Lasso()
#model = linear_model.Ridge()
#model = linear_model.Ridge()
scores = cross_val_score(model, X, y.values.flatten(), cv=cvStrategy, scoring=scoring)
print(scores)
print(np.average(scores))
