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

#Feature Selection & Engineering
#initialize new array 
five_best = []
df_5 = pd.DataFrame(result.corr()["price"]).sort_values(by = "price", ascending = False)
df_5 = df_5.drop(df_5.index[0]).head(5)
for i in range(len(df_5)):
    five_best.append(df_5.index[i]) 
print(five_best)

#three best
three_best = []
df_3 = pd.DataFrame(result.corr()["price"]).sort_values(by = "price", ascending = False)
df_3 = df_3.drop(df_3.index[0]).head(3)
for i in range(len(df_3)):
    three_best.append(df_3.index[i]) 

print(three_best)

#Machine Learning 
#Regression Models
feature_sets = {
    "full_dataset": result.drop(columns=['price']),
    "three_best": result[three_best],
    "five_best": result[five_best], 
   
}

regression_models = {
    "Ridge" : linear_model.Ridge(random_state = 42),
    "DecisionTree" : tree.DecisionTreeRegressor(random_state = 42, max_depth=6),
    "RandomForest" : RandomForestRegressor(random_state = 42),
    "XGBoost": XGBRegressor(random_state = 42),
    "LGBM": LGBMRegressor(random_state = 42),
    "MLP":  MLPRegressor(random_state = 42),    
}

def make_regression(x_train, y_train, x_test, y_test, model, model_name, verbose=True):

    model.fit(x_train,y_train)
    
    y_predict=model.predict(x_train)
    train_error = mean_squared_error(y_train, y_predict, squared=False)
    
    y_predict =model.predict(x_test)
    test_error = mean_squared_error(y_test, y_predict, squared=False)
    
    y_predict=model.predict(x_train)
    r2 = r2_score(y_train, y_predict)
    
    if verbose:
        print("----Model name = {}-----".format(model_name))
        print("Train error = "'{}'.format(train_error))
        print("Test error = "'{}'.format(test_error))
        print("r2_score = "'{}'.format(r2))
        print("--------------------------------")
    
    trained_model = model
    
    return trained_model, y_predict, train_error, test_error, r2

pred_dict = {
    "regression_model": [],
    "feature_set": [],
    "Train Error": [],
    "Test Error": [],
    "R2" : []
}

for feature_set_name in feature_sets.keys():
    
    feature_set = feature_sets[feature_set_name]
    print("Included columns are {}".format(feature_set_name))
    for model_name in regression_models.keys():        
        
        y = result["price"]
        x = feature_set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    

        trained_model, y_predict, train_error, test_error, r2 = make_regression(x_train, y_train, x_test, y_test, regression_models[model_name], model_name, verbose=True)


        pred_dict["regression_model"].append(model_name)
        pred_dict["feature_set"].append(feature_set_name)
        pred_dict["Train Error"].append(train_error)
        pred_dict["Test Error"].append(test_error)
        pred_dict["R2"].append(r2)




#Model Evaluation

#Grid Search in Random Forest

#Deep Learning 