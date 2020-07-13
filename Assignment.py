#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 22:17:52 2019

@author: aninda


"""
#Import the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Import the training dataset
df_train = pd.read_csv('train.csv')
df_train.head()

#Import the test dataset
df_test = pd.read_csv('test.csv')
df_test.head()

#Check the Correlation of training dataset
df_train[df_train.columns[:-1]].corr()
sns.set()
sns.set(font_scale=1.25)
sns.heatmap(df_train[df_train.columns[:-1]].corr(),annot=True,fmt=".1f")
plt.show()

#Check the missing value in traning dataset
df_train.isnull().sum()

# Graphical representation of the missing values.
x = df_train.columns
y = df_train.isnull().sum()
sns.set()
sns.barplot(x,y)
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            int(height),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=1.5)
ax.set_xlabel("Data Attributes")
ax.set_ylabel("count of missing records for each attribute")
plt.xticks(rotation=90)
plt.show()

# This give you the calulation of the education label. Which category of the education label is how many percentage.
total_len = len(df_train['education'])
percentage_labels_education = (df_train['education'].value_counts()/total_len)*100
percentage_labels_education


# Graphical representation of the education label percentage.
sns.set()
sns.countplot(df_train.education).set_title('Data Distribution')
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            '{:.2f}%'.format(100*(height/total_len)),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=1.5)
ax.set_xlabel("Labels for education attribute")
ax.set_ylabel("Numbers of records")
plt.show()

# This give you the calulation of the previous_year_rating label. Which category of the previous_year_rating label is how many percentage.
total_len = len(df_train['previous_year_rating'])
percentage_labels_previous_year_rating = (df_train['previous_year_rating'].value_counts()/total_len)*100
percentage_labels_previous_year_rating

# Graphical representation of the previous_year_rating label percentage.
sns.set()
sns.countplot(df_train.previous_year_rating).set_title('Data Distribution')
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            '{:.2f}%'.format(100*(height/total_len)),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=1.5)
ax.set_xlabel("Labels for previous_year_rating")
ax.set_ylabel("Numbers of records")
plt.show()

#Check the missing value in test dataset
df_test.isnull().sum()

# Graphical representation of the missing values(test data).
x = df_test.columns
y = df_test.isnull().sum()
sns.set()
sns.barplot(x,y)
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            int(height),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=1.5)
ax.set_xlabel("Data Attributes")
ax.set_ylabel("count of missing records for each attribute")
plt.xticks(rotation=90)
plt.show()

# This give you the calulation of the education label. Which category of the education label is how many percentage.(test data)
total_len = len(df_test['education'])
percentage_labels_education = (df_test['education'].value_counts()/total_len)*100
percentage_labels_education

# Graphical representation of the education label percentage.(test data)
sns.set()
sns.countplot(df_test.education).set_title('Data Distribution')
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            '{:.2f}%'.format(100*(height/total_len)),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=1.5)
ax.set_xlabel("Labels for education attribute")
ax.set_ylabel("Numbers of records")
plt.show()

# This give you the calulation of the previous_year_rating label. Which category of the previous_year_rating label is how many percentage.(test data)
total_len = len(df_test['previous_year_rating'])
percentage_labels_previous_year_rating = (df_test['previous_year_rating'].value_counts()/total_len)*100
percentage_labels_previous_year_rating

# Graphical representation of the previous_year_rating label percentage.(test data)
sns.set()
sns.countplot(df_test.previous_year_rating).set_title('Data Distribution')
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            '{:.2f}%'.format(100*(height/total_len)),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=1.5)
ax.set_xlabel("Labels for previous_year_rating")
ax.set_ylabel("Numbers of records")
plt.show()






# Actual replacement of the missing value using the most frequent value.
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer( strategy='most_frequent')
imp_mean.fit(df_train)
imputed_train_df = imp_mean.transform(df_train)
training_data = pd.DataFrame(imputed_train_df)
training_data.head()
imp_mean.fit(df_test)
imputed_test_df = imp_mean.transform(df_test)
test_data = pd.DataFrame(imputed_test_df)
test_data.head()

#Give the column name and reseting the index value
training_data = pd.DataFrame(
    np.row_stack([training_data.columns, training_data.values]),
    columns=['employee_id','department','region','education','gender','recruitment_channel','no_of_trainings','age','previous_year_rating','length_of_service','KPIs_met>80%','awards_won?','avg_training_score','is_promoted']
)
training_data.head()
train = training_data.drop([training_data.index[0]])
train = train.reset_index(drop=True)
train.head()

test_data = pd.DataFrame(
    np.row_stack([test_data.columns, test_data.values]),
    columns=['employee_id','department','region','education','gender','recruitment_channel','no_of_trainings','age','previous_year_rating','length_of_service','KPIs_met>80%','awards_won?','avg_training_score']
)
test_data.head()
test = test_data.drop([test_data.index[0]])
test = test.reset_index(drop=True)
test.head()

#Check the dataframe after the missing value imputation
train.isnull().sum()

#Check the dataframe after the missing value imputation(test data)
test.isnull().sum()

#After missing value imputation check Which category of the education label is how many percentage. 
total_len = len(train['education'])
percentage_labels_education_after = (train['education'].value_counts()/total_len)*100
percentage_labels_education_after

# After missing value imputation graphical representation of the education label percentage.
sns.set()
sns.countplot(train.education).set_title('Data Distribution')
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            '{:.2f}%'.format(100*(height/total_len)),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=1.5)
ax.set_xlabel("Labels for education")
ax.set_ylabel("Numbers of records")
plt.show()

total_len = len(test['education'])
percentage_labels_education_after_test = (test['education'].value_counts()/total_len)*100
percentage_labels_education_after_test

sns.set()
sns.countplot(test.education).set_title('Data Distribution')
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            '{:.2f}%'.format(100*(height/total_len)),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=1.5)
ax.set_xlabel("Labels for education")
ax.set_ylabel("Numbers of records")
plt.show()

#After missing value imputation check Which category of the previous_year_rating label is how many percentage.
total_len = len(train['previous_year_rating'])
percentage_labels_previous_year_rating_after = (train['previous_year_rating'].value_counts()/total_len)*100
percentage_labels_previous_year_rating_after

# After missing value imputation graphical representation of the previous_year_rating label percentage.
sns.set()
sns.countplot(train.previous_year_rating).set_title('Data Distribution')
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            '{:.2f}%'.format(100*(height/total_len)),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=1.5)
ax.set_xlabel("Labels for previous_year_rating")
ax.set_ylabel("Numbers of records")
plt.show()

total_len = len(test['previous_year_rating'])
percentage_labels_previous_year_rating_after_test = (test['previous_year_rating'].value_counts()/total_len)*100
percentage_labels_previous_year_rating_after_test

sns.set()
sns.countplot(test.previous_year_rating).set_title('Data Distribution')
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            '{:.2f}%'.format(100*(height/total_len)),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=1.5)
ax.set_xlabel("Labels for previous_year_rating")
ax.set_ylabel("Numbers of records")
plt.show()

# This give you the calulation of the department label. Which category of the department label is how many percentage.
total_len = len(train['department'])
percentage_labels_department = (train['department'].value_counts()/total_len)*100
percentage_labels_department

# This give you the calulation of the region label. Which category of the region label is how many percentage.
total_len = len(train['region'])
percentage_labels_region = (train['region'].value_counts()/total_len)*100
percentage_labels_region

# This give you the calulation of the recruitment_channel label. Which category of the recruitment_channel label is how many percentage.
total_len = len(train['recruitment_channel'])
percentage_labels_recruitment_channel = (train['recruitment_channel'].value_counts()/total_len)*100
percentage_labels_recruitment_channel

# This give you the calulation of the gender label. Which category of the gender is how many percentage.
total_len = len(train['gender'])
percentage_labels_gender = (train['gender'].value_counts()/total_len)*100
percentage_labels_gender

train.head()





#Few features in the training data are categorical. This is to convert the categorical feature to numerical feature using onehotencoder.
X = train.iloc[:,:].values
Xtest = test.iloc[:,:].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_train = LabelEncoder()
X[:,1] = labelencoder_train.fit_transform(X[:,1])
X[:,2] = labelencoder_train.fit_transform(X[:,2])
X[:,3] = labelencoder_train.fit_transform(X[:,3])
X[:,4] = labelencoder_train.fit_transform(X[:,4])
X[:,5] = labelencoder_train.fit_transform(X[:,5])
onehotencoder = OneHotEncoder(categorical_features=[1,2,3,4,5])
X = onehotencoder.fit_transform(X).toarray()
z = pd.DataFrame(X)
z.head()

Xtest[:,1] = labelencoder_train.fit_transform(Xtest[:,1])
Xtest[:,2] = labelencoder_train.fit_transform(Xtest[:,2])
Xtest[:,3] = labelencoder_train.fit_transform(Xtest[:,3])
Xtest[:,4] = labelencoder_train.fit_transform(Xtest[:,4])
Xtest[:,5] = labelencoder_train.fit_transform(Xtest[:,5])
Xtest = onehotencoder.fit_transform(Xtest).toarray()
ztest = pd.DataFrame(Xtest)
ztest.head()



#Separate the target variable for training
X = z.iloc[:,0:59]
y = z.iloc[:,-1]

#Spliting the dataset for training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#Count of the target variable. How many data point belongs to which class.
unique,count = np.unique(y_train,return_counts = True)
y_train_dict_value_count = { k:v for (k,v) in zip(unique, count)}
y_train_dict_value_count

#The target variable looks like imbalance class. For that using SMOTE sampling technique to balance the class of target variable.
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 12, ratio = 1.0)
X_train_res, y_train_res = sm.fit_sample(X_train,y_train)


#Count of the target variable after sampling technique.
unique,count = np.unique(y_train_res,return_counts = True)
y_train_smote_value_count = { k:v for (k,v) in zip(unique, count)}
y_train_smote_value_count

#Build the model using Logistic regression.
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
glmMod = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                            intercept_scaling=1, class_weight=None, 
                            random_state=None, solver='liblinear', max_iter=100,
                            multi_class='ovr', verbose=2)

glmMod.fit(X_train_res,y_train_res)
glmMod.score(X_test, y_test)
y_pred_logistic_regression = glmMod.predict(X_test)




#Check the accuracy for the model.
result_logistic_regression = classification_report(y_test,y_pred_logistic_regression)
print(result_logistic_regression)

#Hyperparameter optimization using RandomizedSearchCV.
from sklearn.model_selection import RandomizedSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]} 

random_cv = RandomizedSearchCV(LogisticRegression(penalty='l2'),
                              param_distributions=param_grid,
                              cv=10, random_state=42)

random_cv.fit(X_train_res, y_train_res)

random_cv.best_score_
random_cv.best_estimator_

lr_clf = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)

lr_clf.fit(X_train_res, y_train_res)

y_pred = lr_clf.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred_logistic_regression_test = lr_clf.predict(ztest)
y_pred_logistic_regression_test = pd.DataFrame(y_pred_logistic_regression_test)

#Save the test data in a CSV file
df_test['is_promoted'] = y_pred_logistic_regression_test
df_test.to_csv("Logistic_regression.csv")


#Build the model using Random Forest.
from sklearn.ensemble import RandomForestClassifier
rfMod = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=None, min_samples_split=2,
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                               max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, 
                               random_state=None, verbose=0)

rfMod.fit(X_train_res, y_train_res)
rfMod.score(X_test, y_test)
y_pred = rfMod.predict(X_test)

#Check the accuracy for the model.
result_random_forest = classification_report(y_test,y_pred)
print(result_random_forest)

#Hyperparameter optimization using RandomizedSearchCV.
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(1, 45, num = 3)]
min_samples_split = [5, 10]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}

pprint(random_grid)

rf_random = RandomizedSearchCV(rfMod, param_distributions = random_grid, n_iter = 10, cv = 10, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X_train_res, y_train_res)

#Check for the best score and best estimator
rf_random.best_score_
rf_random.best_estimator_

#Fit the model with the best parameters.
rfMod_hyp = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=23, max_features='sqrt', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=10,
                       min_weight_fraction_leaf=0.0, n_estimators=155, n_jobs=1,
                       oob_score=False, random_state=None, verbose=0,
                       warm_start=False)

rfMod_hyp.fit(X_train_res,y_train_res)
rfMod_hyp.score(X_test,y_test)
y_pred = rfMod.predict(X_test)
y_pred_random_forest_test = rfMod_hyp.predict(ztest)
y_pred_random_forest_test = pd.DataFrame(y_pred_random_forest_test)

#Check the accuracy of the model
result_random_forest_hyp = classification_report(y_test,y_pred)
print(result_random_forest_hyp)
confusion_matrix(y_test,y_pred)

#Save the test data in a CSV file
df_test['is_promoted'] = y_pred_random_forest_test
df_test.to_csv("Random_forest.csv")

#Build the model using Gradient Boost.
from sklearn.ensemble import GradientBoostingClassifier
gbMod = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200, subsample=1.0,
                                   min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                   max_depth=3,
                                   init=None, random_state=None, max_features=None, verbose=0)
gbMod.fit(X_train_res, y_train_res)

gbMod.score(X_test, y_test)
y_pred = gbMod.predict(X_test)
y_pred_gradient_boost_test = gbMod.predict(ztest)
y_pred_gredient_boost_test = pd.DataFrame(y_pred_gradient_boost_test)


#Check the accuracy for the model.
result_gradient_boost = classification_report(y_test,y_pred)
print(result_gradient_boost)

#Save the test data in a CSV file
df_test['is_promoted'] = y_pred_gradient_boost_test
df_test.to_csv("Gradient_boost.csv")


