# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:30:58 2024

@author: Admin
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import os

# Load the dataset
df = pd.read_csv("C&T train dataset.csv")
df_test = pd.read_csv("C&T test dataset.csv")

df.info()

null_cols = df.columns[df.isnull().any()]
null_cols

null_cols_test = df_test.columns[df_test.isnull().any()]
null_cols_test

for col in null_cols:
  # Get data type for each column and print it
  data_type = df.dtypes[col]
  print(f"Column '{col}': {data_type}")
  
  
for col in null_cols:
  # Get data type for each column and print it
    print(df[col].value_counts())
    
#df['age'] = df['age'].fillna(round(df['age'].mean()))

null_cols = ['employment_st', 'poi', 'gurantors', 'housing_type','age']


for col in null_cols:
    mode_value = df[col].mode().iloc[0]

    # Replace null values with the mode value
    df[col] = df[col].fillna(mode_value)
    df_test[col] = df_test[col].fillna(mode_value)
    
    
df.isnull().sum()
df_test.isnull().sum()


object_cols = df.select_dtypes(include='object').columns
object_cols

object_cols_test = df_test.select_dtypes(include='object').columns
object_cols_test


# Chi-square test
cols_to_be_kept_object = []
for i in object_cols:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df['Group_no']))
    print(i, '---', pval)
    if pval<0.08:
        cols_to_be_kept_object.append(i)

#removed columns "installment_type" and "employment_st"
    
# VIF for numerical columns
numeric_columns = []
for i in df.columns:
    if df[i].dtype != 'object' and i not in ['sno','Group_no']:
        numeric_columns.append(i)


# VIF sequentially check
vif_data = df[numeric_columns]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0


for i in range (0,total_columns):
    
    vif_value = variance_inflation_factor(vif_data, column_index)
    print (column_index,'---',vif_value)
    
    
    if vif_value <= 7:
        columns_to_be_kept.append( numeric_columns[i] )
        column_index = column_index+1
    
    else:
        vif_data = vif_data.drop([ numeric_columns[i] ] , axis=1)



# check Anova for columns_to_be_kept 

from scipy.stats import f_oneway

columns_to_be_kept_numerical = []

for i in columns_to_be_kept:
    a = list(df[i])  
    b = list(df['Group_no'])  
    
    group_1 = [value for value, group in zip(a, b) if group == 1]
    group_2 = [value for value, group in zip(a, b) if group == 2]
    group_3 = [value for value, group in zip(a, b) if group == 3]


    f_statistic, p_value = f_oneway(group_1, group_2, group_3)

    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)



# listing all the final features
# features = columns_to_be_kept_numerical +object_cols_to_be_kept
# df = df[features + ['Group_no']]


# from sklearn.preprocessing import LabelEncoder

# Fit the LabelEncoder on the training labels
# encoder = LabelEncoder()
# for col in object_cols_to_be_kept:
#   df[f'encoded_{col}'] = encoder.fit_transform(df[col])
# df.drop(object_cols_to_be_kept, axis=1, inplace=True)





from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[columns_to_be_kept_numerical])
df_scaled_test = scaler.fit_transform(df_test[columns_to_be_kept_numerical])

# Assuming df_scaled is a NumPy array
df_scaled = pd.DataFrame(df_scaled,columns=columns_to_be_kept_numerical)
df_scaled_test = pd.DataFrame(df_scaled_test,columns=columns_to_be_kept_numerical)


# listing all the final features
df_target = df[['Group_no']]
df_object = df[cols_to_be_kept_object]
df_object_test = df_test[cols_to_be_kept_object]


# Concatenate now
df = pd.concat([df_object, df_scaled], axis=1)
df_test = pd.concat([df_object_test, df_scaled_test], axis=1)


for i in cols_to_be_kept_object:
    print(df[i].value_counts())


df_encoded = pd.get_dummies(df, columns=cols_to_be_kept_object, drop_first=True)
df_encoded = pd.concat([df_encoded, df_target], axis=1)
df_encoded_test = pd.get_dummies(df_test, columns=cols_to_be_kept_object, drop_first=True)


df_encoded.info()
k = df_encoded.describe()



# Machine Learing model fitting

# Data processing

# 1. Random Forest

y = df_encoded['Group_no']
x = df_encoded. drop ( ['Group_no'], axis = 1 )


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators = 200, random_state=42)
rf_classifier.fit(x_train, y_train)

y_pred = rf_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print ()
print(f'Accuracy: {accuracy}')
print ()
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)


for i, v in enumerate([1,2,3]):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()
# With the help of random forest we got accuracy of 63.75%

# 2. xgboost

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax',  num_class=3)

y = df_encoded['Group_no']
x = df_encoded. drop ( ['Group_no'], axis = 1 )

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

xgb_classifier.fit(x_train, y_train)
y_pred = xgb_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print ()
print(f'Accuracy: {accuracy:.2f}')
print ()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate([1,2,3]):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()
# With the help of random forest we got accuracy of 62.00%
  
    
# 3. Decision Tree
from sklearn.tree import DecisionTreeClassifier

y = df_encoded['Group_no']
x = df_encoded. drop ( ['Group_no'], axis = 1 )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
dt_model.fit(x_train, y_train)
y_pred = dt_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print ()
print(f"Accuracy: {accuracy:.2f}")
print ()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate([1,2,3]):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()
# With the help of decision tree we got accuracy of 61.00%



# Hyper parameter tuning for random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

y = df_encoded['Group_no']
x = df_encoded. drop ( ['Group_no'], axis = 1 )


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators = 200, random_state=42)
rf_classifier.fit(x_train, y_train)

y_pred = rf_classifier.predict(x_test)


# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 8, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# Create the Random Forest model
model = RandomForestClassifier()

# Set up GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Use the best model for prediction
y_pred = best_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print ()
print(f'Accuracy: {accuracy}')
print ()
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)


for i, v in enumerate([1,2,3]):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()

# got accuracy of 65 with best parameters





from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

y = df_encoded['Group_no']
x = df_encoded. drop ( ['Group_no'], axis = 1 )

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)


# Define the hyperparameter grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 8, 12],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3],
    'subsample': [0.5, 0.7, 0.9],
    'colsample_bytree': [0.5, 0.7, 0.9]
}


# Define the XGBClassifier with the initial set of hyperparameters
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3)


grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model with the best hyperparameters on the test set
best_model = grid_search.best_estimator_
accuracy = best_model.score(x_test, y_test)
print("Test Accuracy:", accuracy)


# Use the best model for prediction
y_pred = best_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print ()
print(f'Accuracy: {accuracy}')
print ()
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)


for i, v in enumerate([1,2,3]):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()

Best_Hyperparameters = {'colsample_bytree': 0.9, 
                        'gamma': 0.3, 
                        'learning_rate': 0.1, 
                        'max_depth': 4, 
                        'min_child_weight': 3, 
                        'n_estimators': 200, 
                        'subsample': 0.9
                        }
#Test Accuracy: 0.66875
#Accuracy: 0.66875



