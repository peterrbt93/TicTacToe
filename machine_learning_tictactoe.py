# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 13:55:11 2020

@author: pete_
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load

#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
#from sklearn.cluster import KMeans

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
#from sklearn.pipeline import Pipeline

#from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import learning_curve


#from sklearn.datasets import load_diabetes
#from sklearn.datasets import make_regression
#from sklearn.datasets import make_circles
#from sklearn.datasets import load_breast_cancer
#from sklearn.datasets import make_classification
#from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
#from sklearn.datasets import make_blobs


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix 


#from sklearn import tree



#%%



#Get data from openML as a sklearn.utils.bunch object
data = fetch_openml('tic-tac-toe')#, return_X_y=True)

#Print properties
print("Features and their possible values are")
print(data.categories)
print("Target names")
print(data.target_names)

#From this we learn that features are values for the positions of the board, 
# and target is positive if player X wins

#From description we now know that "positive" targets refer to play x winning.

def data_to_board(x):
    """
    Parameters
    ----------
    x : Data example

    Returns
    -------
    A numpy 3x3 array showing the corresponding board
    
    """
    result = np.where(x==2.0,'x',x)
    result = np.where(x==1.0,'o',result)
    result = np.where(x==0.0,' ',result)
    return result.reshape(3,3)            
    
#Assign X,y from data
X,y = data.data, data.target

#convert y to boolean integers
y = np.where(y=='positive',1,y)
y = np.where(y=='negative',0,y)
y=y.astype('int')


#Make copy of old data in order to print board
X_copy = X

#Convert X to Onehot (categories instead of number values)
enc = OneHotEncoder()
enc.fit(X)
X = enc.transform(X).toarray()

#Info from encoder
#enc.categories_,enc.inverse_transform,enc.get_feature_names


#%%

#Try different model on a CV=5 split

poly = PolynomialFeatures(degree = 2, interaction_only=False, include_bias=False)


kf = KFold(n_splits=5, shuffle=True, random_state=1)
lr_scores = []
#lrp_scores = []
dt_scores = []
rf_scores = []
svm_scores = []
mlp_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    

    lr = LogisticRegression(solver='lbfgs',max_iter=1000)
    lr.fit(X_train, y_train)
    lr_scores.append(lr.score(X_test, y_test))
    
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_scores.append(dt.score(X_test, y_test))
    
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_scores.append(rf.score(X_test, y_test))
    
    svm = SVC()
    svm.fit(X_train, y_train)
    svm_scores.append(svm.score(X_test, y_test))
    
    mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50), alpha=0.01, solver='adam', random_state=3)
    mlp.fit(X_train, y_train)
    mlp_scores.append(mlp.score(X_test, y_test))
    
    #or for polynommial LR -> LRP
   # X_poly_train = poly.fit_transform(X_train)
   # X_poly_test = poly.fit_transform(X_test)
   # lrp = LogisticRegression(solver='lbfgs')
   # lrp.fit(X_poly_train, y_train)
   # lrp_scores.append(lrp.score(X_poly_test, y_test))
    
    
    
print("LR accuracy:", round(np.mean(lr_scores),2))
#print("LRP accuracy:", round(np.mean(lrp_scores),2))
print("DT accuracy:", round(np.mean(dt_scores),2))
print("RF accuracy:", round(np.mean(rf_scores),2))
print("SVM accuracy:", round(np.mean(svm_scores),2))
print("MLP accuracy:", round(np.mean(mlp_scores),2),"\n")


#OMG had to do onehot preprocessing first! xD

#%%

#After grid/bayes search the estimator is initialized with
# the best parameters

#LR

strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)


#GridSearch
lr_param_grid = {
    'C': [0.001,0.1, 1,20,200,500],
    'solver': ['newton-cg', 'lbfgs', 'liblinear']}

lr = LogisticRegression(max_iter=150)
lr_gs = GridSearchCV(lr, lr_param_grid, scoring='accuracy', cv=strat_k_fold)
lr_gs.fit(X, y)
print("best GS test score:", lr_gs.best_score_) 
print("best GS total score:", lr_gs.score(X,y)) 
print("best GS params:",lr_gs.best_params_)

#BayesSearch
lr_param_grid = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'solver': Categorical(['newton-cg', 'lbfgs', 'liblinear'])}

lr = LogisticRegression(max_iter=150)
lr_bs = BayesSearchCV(lr, lr_param_grid, scoring='accuracy', cv=strat_k_fold,n_iter=20)
lr_bs.fit(X, y)
print("best BS test score:", lr_bs.best_score_) 
print("best BS total score:", lr_bs.score(X,y)) 
print("best BS params:",lr_bs.best_params_)


#%%


#RF

strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)


#GridSearch
rf_param_grid = {
    'n_estimators': [5, 15, 50,100],
    'max_depth': [5, 15, 25],
    'min_samples_leaf': [1, 3],
    'max_leaf_nodes': [10, 20, 50, 100]}

rf = RandomForestClassifier(random_state=1)
rf_gs = GridSearchCV(rf, rf_param_grid, scoring='accuracy', cv=strat_k_fold)
rf_gs.fit(X, y)
print("best GS test score:", rf_gs.best_score_) 
print("best GS total score:", rf_gs.score(X,y)) 
print("best GS params:",rf_gs.best_params_)

#BayesSearch
rf_param_grid = {
    'n_estimators': Integer(5,100),
    'max_depth': Integer(5,25),
    'min_samples_leaf': Integer(1,3),
    'max_leaf_nodes': Integer(10,100)}

rf = RandomForestClassifier()
rf_bs = BayesSearchCV(rf, rf_param_grid, scoring='accuracy', cv=strat_k_fold,n_iter=40)
rf_bs.fit(X, y)
print("best BS test score:", rf_bs.best_score_) 
print("best BS total score:", rf_bs.score(X,y)) 
print("best BS params:",rf_bs.best_params_)    



#%%
#SVM

strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)


#GridSearch
svm_param_grid = {
    'C': [0.1, 1,20,200,500],
    'kernel': ['linear','poly', 'rbf','sigmoid'],
    'degree': [1, 3, 5]}

svm = SVC()
svm_gs = GridSearchCV(svm, svm_param_grid, scoring='accuracy', cv=strat_k_fold)

svm_gs.fit(X, y)
print("best CS test score:", svm_gs.best_score_) 
print("best CS total score:", svm_gs.score(X,y)) 
print("best CS params:",svm_gs.best_params_)
    
#BayesSearch
svm_param_grid = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'kernel': Categorical(['linear', 'poly', 'rbf','sigmoid']),
    'degree': Integer(1,5)}

svm = SVC()
svm_bs = BayesSearchCV(svm, svm_param_grid, scoring='accuracy', cv=strat_k_fold,n_iter=10)

svm_bs.fit(X, y)
print("best BS test score:", svm_bs.best_score_) 
print("best BS total score:", svm_bs.score(X,y)) 
print("best BS params:",svm_bs.best_params_)


#%%

#MLP

strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)


#GridSearch
mlp_param_grid = {
    'hidden_layer_sizes': [(100,), (50,)],
    'activation': ['relu','logistic'],
    'solver': ['adam','lbfgs'],
    'alpha': [0.01,0.1,0.5],
    'learning_rate': ['adaptive']}

mlp = MLPClassifier(max_iter=2000)
mlp_gs = GridSearchCV(mlp, mlp_param_grid, scoring='accuracy', cv=strat_k_fold)

mlp_gs.fit(X, y)
print("best GS test score:", mlp_gs.best_score_) 
print("best GS total score:", mlp_gs.score(X,y)) 
print("best GS params:",mlp_gs.best_params_)

#BayesSearch

#Bayes grid cant do tuples yet...
mlp_param_grid = {
    'hidden_layer_sizes': Categorical([50,100,200]),
    'activation': Categorical(['relu','logistic']),
    'solver': Categorical(['adam','lbfgs']),
    'alpha': Real(1e-2, 1e+1, prior='log-uniform'),
    'learning_rate': Categorical(['adaptive'])}

mlp = MLPClassifier(max_iter=2000)
mlp_bs = BayesSearchCV(mlp, mlp_param_grid, scoring='accuracy', cv=strat_k_fold,n_iter=10)

mlp_bs.fit(X, y)
print("best BS test score:", mlp_bs.best_score_) 
print("best BS total score:", mlp_bs.score(X,y)) 
print("best BS params:",mlp_bs.best_params_)

#%%

#Final evaluation
print("\n For LR")
print("best GS test score:", lr_gs.best_score_) 
print("best GS total score:", lr_gs.score(X,y)) 
print("best GS params:",lr_gs.best_params_)
print("best BS test score:", lr_bs.best_score_) 
print("best BS total score:", lr_bs.score(X,y)) 
print("best BS params:",lr_bs.best_params_)

print("\n For RF")
print("best GS test score:", rf_gs.best_score_) 
print("best GS total score:", rf_gs.score(X,y)) 
print("best GS params:",rf_gs.best_params_)
print("best BS test score:", rf_bs.best_score_) 
print("best BS total score:", rf_bs.score(X,y)) 
print("best BS params:",rf_bs.best_params_)

print("\n For SVM")
print("best GS test score:", svm_gs.best_score_) 
print("best GS total score:", svm_gs.score(X,y)) 
print("best GS params:",svm_gs.best_params_)
print("best BS test score:", svm_bs.best_score_) 
print("best BS total score:", svm_bs.score(X,y)) 
print("best BS params:",svm_bs.best_params_)

print("\n For MLP")
print("best GS test score:", mlp_gs.best_score_) 
print("best GS total score:", mlp_gs.score(X,y)) 
print("best GS params:",mlp_gs.best_params_)
print("best BS test score:", mlp_bs.best_score_) 
print("best BS total score:", mlp_bs.score(X,y)) 
print("best BS params:",mlp_bs.best_params_)


#%%

#Tweaking best classifier

#The model that performed best (speed was also important) 
# was SVM with GridSearch (Bayes time depended
#on the random_state which was not so good)! 
# now, we will try
#To tweak the parameter space to see if we can do better
#still! This is done progressively

strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)


#GridSearch
svm_param_grid = {
    'C': [0.1, 1,20,200,500,5000],
    'kernel': ['poly', 'rbf'],
    'degree': [1, 3, 4]}

svm = SVC()
svm_gs = GridSearchCV(svm, svm_param_grid, scoring='accuracy', cv=strat_k_fold)

svm_gs.fit(X, y)
print("best CS test score:", svm_gs.best_score_) 
print("best CS total score:", svm_gs.score(X,y)) 
print("best CS params:",svm_gs.best_params_)



#%%

#Final model

model = SVC(C=1,degree=4,kernel='poly')

strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

#CV score
print("CV final score:",np.mean(cross_val_score(model,X,y,cv=strat_k_fold)))

#Fit data now in order to predict on new stuff
model.fit(X,y)
print('Score on total set:',model.score(X,y))
#Should be 100% since it has trained on the entire set
print('-----------------------')


#Examples
for n in [0,100,750,700]:
    print(data_to_board(X_copy[n]))
    print("Who wins?: ","x" if model.predict([X[n]])==[1] else "o" )
    print('-----------------------')
    
#Save trained model
dump(model, 'fittedmodel.joblib')


#Load trained model
model = load('fittedmodel.joblib')