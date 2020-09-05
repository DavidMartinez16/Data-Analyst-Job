# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 10:13:12 2020

@author: david
"""

# ----------------------- IMPORT LIBRARIES --------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# ------------------------ PREPARE THE DATA ----------------------------------
# Read the dataset
df = pd.read_csv('Cleaned_Data.csv')

df.columns

# Select the features
df_model = df[['avg_salary','Job Title','Rating','Size','Type of ownership','Industry', 
               'Sector', 'Revenue','loc','age', 'Python', 'Excel',
               'Matlab', 'Spark', 'aws',]]

# Get dummies of data
df_dum = pd.get_dummies(df_model)

# Select the variables
x = np.array(df_dum.drop(['avg_salary'],axis=1))
y = np.array(df['avg_salary'])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

# --------------------------------- MODELS BUILDING --------------------------------

# Linear Regression
lr = LinearRegression()
lr.fit(x_train,y_train)
lr_cv = np.mean(cross_val_score(lr,x_train,y_train,cv=5,scoring='neg_mean_absolute_error'))

# Random Forest
rf = RandomForestRegressor()
rf.fit(x_train,y_train)
rf_cv = np.mean(cross_val_score(rf,x_train,y_train,cv=5,scoring='neg_mean_absolute_error'))

# K-Nearest Neighbors
knn = KNeighborsRegressor()
knn.fit(x_train,y_train)
knn_cv = np.mean(cross_val_score(knn,x_train,y_train,cv=5,scoring='neg_mean_absolute_error'))

# Bagging Regressor
br = BaggingRegressor()
br.fit(x_train,y_train)
br_cv = np.mean(cross_val_score(br,x_train,y_train,cv=5,scoring='neg_mean_absolute_error'))

# According to the mean absolute error in cross validation the model with
# the lowest error is Random Forest

# ------------------------- TUNNING THE SELECTED MODEL ----------------------

parameters = {'n_estimators':range(100,500,100), 'criterion':('mse','mae'),'max_features':('auto','sqrt','log2')}
clf = GridSearchCV(rf,parameters)
clf.fit(x_train,y_train)

# Get the best score and estimator
clf.best_score_
clf.best_estimator_

# Make the prediction with the Random Forest Model
y_pred_rf = rf.predict(x_test)

# MAE
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Top 8 Best Features by Random Forest
importances = rf.feature_importances_[:8]
plt.title('Feature Importance',weight='bold',fontsize=20)
sns.barplot(data={'importance':importances,'feature':df_dum.columns[df_dum.columns!='avg_salary'][:8]},y='feature',x='importance')
plt.show()
