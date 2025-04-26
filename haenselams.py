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
    train_error = mean_squared_error(y_train, y_predict)
    
    y_predict =model.predict(x_test)
    test_error = mean_squared_error(y_test, y_predict)
    
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

pred_df = pd.DataFrame(pred_dict)
print(pred_df.head(5))

pred_df["feature_set_2"] = pred_df["feature_set"].apply(lambda x: x.split('_')[0])

pred_df["Model_with_Data_set"] = pred_df['regression_model'] +"_"+ pred_df["feature_set_2"]

df_barh = pred_df[["Train Error","Test Error", "R2", "Model_with_Data_set" ]]

df_train_error = df_barh[['Model_with_Data_set', 'Train Error']]
df_test_error = df_barh[['Model_with_Data_set', 'Test Error']]

# Create a figure and subplots
fig, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(14, 6))


# Create the first graph
df_barh.plot(kind='barh', x='Model_with_Data_set', y='R2', color='red', ax=ax2, legend=False)
ax2.set_xlabel('R Squared')
ax2.set_ylabel('Model')
ax2.set_title('R-squared')

# Create the second graph
df_train_error.plot(kind='barh', x='Model_with_Data_set', y='Train Error', color='blue', ax=ax3, legend=False)
ax3.set_xlabel('Train Error')
ax3.set_ylabel('Model')
ax3.set_title('Train Error')


# Create the second graph
df_test_error.plot(kind='barh', x='Model_with_Data_set', y='Test Error', color='green', ax=ax4, legend=False)
ax4.set_xlabel('Test Error')
ax4.set_ylabel('Model')
ax4.set_title('Test Error')

# Fit the figure
plt.tight_layout()

# Show the figure
plt.show()

pred_df.drop(columns=['feature_set_2', 'Model_with_Data_set'], inplace=True)

#Model Evaluation - Highest R Squared, Min test error, Min train error,

#Highest R Squared
print(pred_df.sort_values(by = "R2", ascending = False).head(5))

#min test error 
print(pred_df.sort_values(by = "Test Error", ascending = True).head(5))

#min train error 
print(pred_df.sort_values(by = "Train Error", ascending = True).head(5))

#Grid Search in Random Forest
'''

x = result.drop(columns=['price'])
y = result["price"]

x_train, x_test, y_train, y_test = train_test_split(
x, y, 
test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)

model_name = "Random Forest"

model.fit(x_train,y_train)

y_predict=model.predict(x_train)
train_error_f = mean_squared_error(y_train, y_predict)

y_predict =model.predict(x_test)
test_error_f = mean_squared_error(y_test, y_predict)

y_predict=model.predict(x_train)
r2_f = r2_score(y_train, y_predict)


print("----Model name = {}-----".format(model_name))
print("Train error = "'{}'.format(train_error_f))
print("Test error = "'{}'.format(test_error_f))
print("r2_score = "'{}'.format(r2_f))
print("--------------------------------")

print(model.get_params())

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    
    'bootstrap': [True],
    'ccp_alpha' : [0.0],
    'criterion': ['squared_error'],
    'max_depth': [None],    
    'max_features':[sqrt(n_features)],
    'max_leaf_nodes' : [None],
    'max_samples' : [None],
    'min_impurity_decrease' : [0.0],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'min_weight_fraction_leaf' : [0.0],
    'n_estimators': [50, 500, 700,],
    'n_jobs' : [None],
    'oob_score' :[True, False],
    'random_state' : [42],
    'verbose' : [0],
    'warm_start' : [True, False],
    
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 0, scoring = 'neg_mean_squared_error')

import time
start = time.time()
grid_search.fit(x_train, y_train)

end = time.time()
elapsed_time_seconds = end - start
elapsed_time = (elapsed_time_seconds) / 60
elapsed_time_seconds = round(elapsed_time_seconds,2)
elapsed_time = round(elapsed_time,2)
print('Execution time:', elapsed_time_seconds, 'seconds which is:', elapsed_time, 'minutes.' )

best_params = grid_search.best_params_

x = result.drop(columns=['price'])
y = result["price"]
x_train, x_test, y_train, y_test = train_test_split(
x, y, 
test_size=0.2, random_state=42
)

model = RandomForestRegressor()

model.set_params(**best_params)

model_name = "Random Forest"

model.fit(x_train,y_train)

y_predict= model.predict(x_train)
train_error_gr = mean_squared_error(y_train, y_predict, squared=False)

y_predict =model.predict(x_test)
test_error_gr = mean_squared_error(y_test, y_predict, squared=False)

y_predict=model.predict(x_train)
r2_gr = r2_score(y_train, y_predict)


print("----Model name = {}-----".format(model_name))
print("Train error = "'{}'.format(train_error_gr))
print("Test error = "'{}'.format(test_error_gr))
print("r2_score = "'{}'.format(r2_gr))
print("--------------------------------")


#Scaling 

#Min-Max Scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

x = result.drop(columns=['price'])
y = result["price"]
y = np.asanyarray(y).reshape(-1,1)

x_scaler = MinMaxScaler()
x_scaler.fit(X=x, y=y)
x_scaled = x_scaler.transform(x)
y_scaler = MinMaxScaler()
y_scaler.fit(y)
y_scaled = y_scaler.transform(y)

x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
        x_scaled, y_scaled, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.set_params(**best_params)
model.fit(x_train_scaled,y_train_scaled)


y_predict_scaled=model.predict(x_train_scaled)
y_predict_train_minmax = y_scaler.inverse_transform(y_predict_scaled.reshape(-1, 1))
rmse_error_train_minmax = mean_squared_error(y_train, y_predict_train_minmax[:,0], squared=False)
r2_mm = r2_score(y_train, y_predict_train_minmax)

model = RandomForestRegressor()
model.set_params(**best_params)
model.fit(x_test_scaled,y_test_scaled)


y_predict_scaled_t=model.predict(x_test_scaled)
y_predict_test_minmax = y_scaler.inverse_transform(y_predict_scaled_t.reshape(-1, 1))
rmse_error_test_minmax = mean_squared_error(y_test, y_predict_test_minmax[:,0], squared=False)

print("Normalized Train error = "'{}'.format(rmse_error_train_minmax))
print("Normalized Test error = "'{}'.format(rmse_error_test_minmax))
print("Normalized r2_score = "'{}'.format(r2_mm))

#Standard Scaling
from sklearn.preprocessing import StandardScaler

x_scaler = StandardScaler()
x_scaler.fit(X=x, y=y)
x_scaled = x_scaler.transform(x)
y_scaler = StandardScaler()
y_scaler.fit(y)
y_scaled = y_scaler.transform(y)


x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
        x_scaled, y_scaled, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.set_params(**best_params)
model.fit(x_train_scaled,y_train_scaled)

y_predict_scaled=model.predict(x_train_scaled)
y_predict_train_standard = y_scaler.inverse_transform(y_predict_scaled.reshape(-1, 1))
rmse_error_train_standard = mean_squared_error(y_train, y_predict_train_standard[:,0], squared=False)
r2_standard = r2_score(y_train, y_predict_train_standard)

model = RandomForestRegressor()
model.set_params(**best_params)
model.fit(x_test_scaled,y_test_scaled)


y_predict_scaled_t=model.predict(x_test_scaled)
y_predict_test_standard = y_scaler.inverse_transform(y_predict_scaled_t.reshape(-1, 1))
rmse_error_test_standard = mean_squared_error(y_test, y_predict_test_standard[:,0], squared=False)

print("Normalized Train error = "'{}'.format(rmse_error_train_standard))
print("Normalized Test error = "'{}'.format(rmse_error_test_standard))
print("Normalized r2_score = "'{}'.format(r2_standard))


#Robust Scaling
from sklearn.preprocessing import RobustScaler

x_scaler = RobustScaler()
x_scaler.fit(X=x, y=y)
x_scaled = x_scaler.transform(x)
y_scaler = RobustScaler()
y_scaler.fit(y)
y_scaled = y_scaler.transform(y)


x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
        x_scaled, y_scaled, test_size=0.2, random_state=42)


model = RandomForestRegressor()
model.set_params(**best_params)
model.fit(x_train_scaled,y_train_scaled)

y_predict_scaled=model.predict(x_train_scaled)
y_predict_train_robust = y_scaler.inverse_transform(y_predict_scaled.reshape(-1, 1))
rmse_error_train_robust = mean_squared_error(y_train, y_predict_train_robust[:,0], squared=False)
r2_robust = r2_score(y_train, y_predict_train_robust)

model = RandomForestRegressor()
model.set_params(**best_params)
model.fit(x_test_scaled,y_test_scaled)

y_predict_scaled_t=model.predict(x_test_scaled)
y_predict_test_robust = y_scaler.inverse_transform(y_predict_scaled_t.reshape(-1, 1))
rmse_error_test_robust = mean_squared_error(y_test, y_predict_test_robust[:,0], squared=False)

print("Normalized Train error = "'{}'.format(rmse_error_train_robust))
print("Normalized Test error = "'{}'.format(rmse_error_test_robust))
print("Normalized Train r2_score = "'{}'.format(r2_robust))

pred_dict = {
    "technique": ["Base Model", "Grid Search", "Min-Max Scaling", "Standard-Scaling", "Robust-Scaling"], #min-max scaling, #standard scaling, #robust # grid search
    "Train Error": [train_error_f, train_error_gr, rmse_error_train_minmax, rmse_error_train_standard, rmse_error_train_robust],
    "Test Error": [test_error_f, test_error_gr, rmse_error_test_minmax, rmse_error_test_standard, rmse_error_test_robust],
    "R2" : [r2_f, r2_gr, r2_mm, r2_standard, r2_robust]
}

pred_df = pd.DataFrame(pred_dict)

print(pred_df)

# Create a figure and subplots
fig, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(14, 6))

# Create the first graph
pred_df.plot(kind='bar', x='technique', y='R2', color='red', ax=ax2, legend=False)
ax2.set_xlabel('R Squared')
ax2.set_title('R-squared')
ax2.tick_params(axis='x', labelrotation=45)





# Create the second graph
pred_df.plot(kind='bar', x='technique', y='Train Error', color='blue', ax=ax3, legend=False)
ax3.set_xlabel('Train Error')
ax3.set_title('Train Error')
ax3.tick_params(axis='x', labelrotation=45)




# Create the second graph
pred_df.plot(kind='bar', x='technique', y='Test Error', color='green', ax=ax4, legend=False)
ax4.set_xlabel('Test Error')
ax4.set_title('Test Error')
ax4.tick_params(axis='x', labelrotation=45)



# Fit the figure
plt.tight_layout()

# Show the figure
plt.show()
'''

#Deep Learning 

y = result["price"]
x = result.drop(columns=['price'])

x_train, x_test, y_train, y_test = train_test_split(
x, y, 
test_size=0.2, random_state=42
)

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
import time

start = time.time()

tf.random.set_seed(42)

# Define a new model with more layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=[x_train.shape[1]]),
    tf.keras.layers.Dense(units=1)
])

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.losses.MeanSquaredError(),
              metrics=[tf.metrics.MeanAbsoluteError()])


# Fit the model to the training data
history = model.fit(x_train, y_train, epochs=30, batch_size=32, verbose = 1,
                    validation_data=(x_test, y_test))

y_train_pred = model.predict(x_train)
r2_first = r2_score(y_train, y_train_pred)

# Select the MAE and val_MAE for the four desired epochs
epochs_to_plot = [5, 10, 15, 20, 25, 30]
mae_first = [history.history['mean_absolute_error'][epoch - 1] for epoch in epochs_to_plot]
val_mae_first = [history.history['val_mean_absolute_error'][epoch - 1] for epoch in epochs_to_plot]

# Plot the MAE
plt.plot(epochs_to_plot, mae_first, 'b', label=f'Training MAE: {mae_first[-1]:.3f}')

# Plot the val_MAE
plt.plot(epochs_to_plot, val_mae_first, 'r', label=f'Test MAE: {val_mae_first[-1]:.3f}')
plt.legend()
plt.show()

end = time.time()
elapsed_time_seconds = end - start
elapsed_time = (elapsed_time_seconds) / 60
elapsed_time_seconds = round(elapsed_time_seconds,2)
elapsed_time = round(elapsed_time,2)
print('Execution time:', elapsed_time_seconds, 'seconds which is:', elapsed_time, 'minutes.' )


import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time
from tensorflow.keras import regularizers

start = time.time()

tf.random.set_seed(42)

# Define a new model with more layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=[x_train.shape[1]], kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(units=16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(units=8, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(units=1)
])

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
              loss=tf.losses.MeanSquaredError(),
              metrics=[tf.metrics.MeanAbsoluteError()])


# Fit the model to the training data
history = model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose = 0,
                    validation_data=(x_test, y_test))

y_train_pred = model.predict(x_train)
r2_second = r2_score(y_train, y_train_pred)

# Select the MAE and val_MAE for the four desired epochs
epochs_to_plot = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
mae_second = [history.history['mean_absolute_error'][epoch - 1] for epoch in epochs_to_plot]
val_mae_second = [history.history['val_mean_absolute_error'][epoch - 1] for epoch in epochs_to_plot]

# Plot the MAE
plt.plot(epochs_to_plot, mae_second, 'b', label=f'Training MAE: {mae_second[-1]:.3f}')

# Plot the val_MAE
plt.plot(epochs_to_plot, val_mae_second, 'r', label=f'Test MAE: {val_mae_second[-1]:.3f}')
plt.legend()
plt.show()

end = time.time()
elapsed_time_seconds = end - start
elapsed_time = (elapsed_time_seconds) / 60
elapsed_time_seconds = round(elapsed_time_seconds,2)
elapsed_time = round(elapsed_time,2)
print('Execution time:', elapsed_time_seconds, 'seconds which is:', elapsed_time, 'minutes.' )


pred_dict = {
    "Algorithm": ["DL", "DL optimized"], #min-max scaling, #standard scaling, #robust # grid search
    "Train Error": [mae_first[-1] , mae_second[-1]],
    "Test Error": [val_mae_first[-1] , val_mae_second[-1]],
    "R2" : [r2_first, r2_second] }

pred_df = pd.DataFrame(pred_dict)
print(pred_df)


# Create a figure and subplots
fig, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(10, 4))

# Create the first graph
pred_df.plot(kind='bar', x='Algorithm', y='R2', color='red', ax=ax2, legend=False)
ax2.set_xlabel('R Squared')
ax2.set_title('R-squared')
ax2.tick_params(axis='x', labelrotation=45)



# Create the second graph
pred_df.plot(kind='bar', x='Algorithm', y='Train Error', color='blue', ax=ax3, legend=False)
ax3.set_xlabel('Train Error')
ax3.set_title('Train Error')
ax3.tick_params(axis='x', labelrotation=45)


# Create the second graph
pred_df.plot(kind='bar', x='Algorithm', y='Test Error', color='green', ax=ax4, legend=False)
ax4.set_xlabel('Test Error')
ax4.set_title('Test Error')
ax4.tick_params(axis='x', labelrotation=45)