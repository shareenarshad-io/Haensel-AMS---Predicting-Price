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
df.head(5)
df.info()

#Data Manipulation
df["loc1"].value_counts()
df = df[(df["loc1"].str.contains("S") == False) & (df["loc1"].str.contains("T") == False)]
df.shape 
df["loc2"] = pd.to_numeric(df["loc2"], errors='coerce')
df["loc1"] = pd.to_numeric(df["loc1"], errors='coerce')
df.dropna(inplace = True)
df.shape

#Data Type Changing 
dow_dummies = pd.get_dummies(df.dow)
