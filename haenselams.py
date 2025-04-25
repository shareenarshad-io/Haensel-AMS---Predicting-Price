#Haenselams.py contains code and sample.csv includes sample data for this project
import pandas as pd 
import numpy as np 
import matplotlib as pyplot 

#Assignment Instructions:
#Assignment
#The target variable is price. You have 7 attributes and obviously we want you to build some initial ML model which predicts the prices.

#Make some initial data analysis, which will hint to some stuctures in the data and how attributes are connected.
#Fit some ML model(s) and explain briefly your choices.
#Show with some X-validation the power of your model and comment the results.
#Present us the results and the steps you have taken and also with some critical thinking and next steps.

#Data Preparation & Data Exploration:

#other libraries to import
import xgboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor 
import sklearn
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

#Steps for Data Science Project
#1. Problem Definition
#2. Data Collection and Preparation
#3. Exploratory Data Analysis 
#4. Feature Engineering
#5. Model Building
#6. Model Evaluation

#Read in the sample data
df = pd.read_csv("sample.csv")
print(df.head(5))
print(df.info())

#Data Manipulation
print(df["loc1"].value_counts())
#drop all the S & T entries
df = df[(df["loc1"].str.contains("S") == False) & (df["loc1"].str.contains("T") == False)]
print(df.shape) #(9998, 8)
#Convert the loc2 and loc 1 columns to numeric, rather than object, if it can't it gets replaced with NaN
df["loc2"] = pd.to_numeric(df["loc2"], errors='coerce')
df["loc1"] = pd.to_numeric(df["loc1"], errors='coerce')
#Can see changes to numeric values here 
print(df.info())
#drops all NaN values or missing values
df.dropna(inplace = True)
print(df.shape) # (9993, 8) -> dropped 5 rows 


#Data Type Changing 
#create one hot encoding of the days of the week variable 
dow_dummies = pd.get_dummies(df.dow)
dow_dummies.replace({False: 0, True: 1}, inplace=True)

print(dow_dummies.head())
df2 = df.copy(deep=True)
df2.drop(columns = "dow", inplace = True)
result = df2.join(dow_dummies)
print(result.head())

#map to the days of the week and replace column dow
days_of_week = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}
df['dow'] = df['dow'].map(days_of_week)
print(df.head())

#Checking Outliers and Correlations 
from pandas.plotting import scatter_matrix

# Suppress the output of the scatter_matrix function, a way to visualize the relationships between multiple variables in a dataset
_ = scatter_matrix(result.iloc[:,0:7], figsize=(12, 8))

#sorting the data and cleaning up the data more and splitting the data too
print(pd.DataFrame(abs(result.corr()["price"])).sort_values(by = "price", ascending = False)[1::])
print(result.drop(columns = "loc2", inplace = True))
print(result.iloc[:,0:6])

#placing the data into bins for a histogram to analyze
import matplotlib.pyplot as plt 
print(result.iloc[:,0:6].hist(bins=50, figsize=(20,15)))
plt.show()

print(result.iloc[:,0:6].describe())

#sort and describe value counts and have a cut off too 
print(result.iloc[:,0:6].sort_values(by = "para1", ascending = False).head(5))

print(result["para1"].value_counts())
result = result[result["para1"] < 10]

#Feature Selection
#initialize new array 
five_best = []
df_5 = pd.DataFrame(result.corr()["price"]).sort_values(by = "price", ascending = False)
