# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 20:02:15 2021

@author: PC
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:09:03 2021

@author: PC
"""
import pandas as pd
import numpy as np
df_data = pd.read_csv('C:/Meet College/ECMT 673/Project_Meet/aaaaaaa.csv')
df_data.head(10)
data = df_data.dropna()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
df = data
y = df['Sales_Per_capita']
x = scaler.fit_transform(df.iloc[:,0:7])
# importing train_test_split from sklearn
from sklearn.model_selection import train_test_split
# splitting the data
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
# importing module
from sklearn.linear_model import LinearRegression
# creating an object of LinearRegression class
LR = LinearRegression()
# fitting the training data
model = LR.fit(x,y)
y_prediction =  LR.predict(x)
y_prediction
df_pred = pd.DataFrame(y_prediction, columns = ['Sales_Per_capita'])
df_pred.to_csv('pred3.csv')
#importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
score=r2_score(y,y_prediction)
print("r2 socre is ",score)
print("mean_sqrd_error is==",mean_squared_error(y,y_prediction))
print("root_mean_squared error of is==",np.sqrt(mean_squared_error(y,y_prediction)))


'''# linear regression feature importance
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
# get importance
importance = model.coef_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()'''

import seaborn as sns
import matplotlib.pyplot as plt
#get correlations of each features in dataset
df_corr = pd.DataFrame(df, columns = ['Per_Capita_Net_Earnings','Unemp_Rate','laborforceavg','male_20_39avg','tot_maleavg','tot_femaleavg','employment_rate','Sales_Per_capita'])
corrmat = df_corr.corr()
corrmat.to_csv('photo.csv')
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
#g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#df_corr.to_csv('df_norm.csv')