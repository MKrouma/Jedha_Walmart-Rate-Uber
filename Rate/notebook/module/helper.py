""" helper functions for project.
"""

# import
# import
import os
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


# helper functions : general_statistics()
def general_statistics(df) :
    """ general eda.
    """

    # Basic stats
    print("Number of rows : {}".format(df.shape[0]))
    print()

    print("Display of dataset: ")
    display(df.head())
    print()

    print("Basics statistics: ")
    data_desc = df.describe(include='all')
    display(data_desc)
    print()

    print("General information: ")
    display(df.info())
    print()

    print("Percentage of missing values: ")
    display(100*df.isnull().sum()/df.shape[0])

# column indice in df
def var_index(df, name) :
    return df.columns.to_list().index(name)

# MODEL
# modularize : train model
def train_model(model_name, X, y, log=False, params={}) :
    """ train model
    """

    # LR
    if model_name == "LogisticRegression" : 

        # train LR model
        classifier = LogisticRegression() 
        classifier.fit(X, y)

        if log : 
            print(f"trained {model_name} model !")

    # LR
    if model_name == "RandomForest-GridS" : 

        # Perform grid search
        classifier = RandomForestClassifier()

        # Grid of values to be tested
        if bool(params) == True : 
            gridsearch = GridSearchCV(classifier, param_grid = params, cv = 3)
            gridsearch.fit(X, y)

            if log :             
                print(f"trained {model_name} model !")
                print("best hyperparameters : ", gridsearch.best_params_)
                print("best validation accuracy : ", gridsearch.best_score_)

            # fit with best params
            print("\ntraining model with best hyperparameters ...")
            best_params = gridsearch.best_params_
            max_depth = best_params["max_depth"]
            min_samples_leaf = best_params["min_samples_leaf"]
            min_samples_split = best_params["min_samples_split"]
            n_estimators = best_params["n_estimators"]

            classifier = RandomForestClassifier(max_depth=max_depth, 
                                                min_samples_leaf=min_samples_leaf,
                                                min_samples_split=min_samples_split,
                                                n_estimators=n_estimators)
            classifier.fit(X, y)

            best_params = pd.DataFrame(best_params)
            best_params.to_file("../data/temp/{model_name}_bestHyperParam.csv")

        else : 
            return print("Enter grid search parameters.")

        
    # save 
    filename = f'../data/temp/{model_name}.joblib.pkl'
    _ = joblib.dump(classifier, filename, compress=9)

    return filename

# prediction
def prediction(model, X, step="", log=True) :
    """ model prediction on X.
    """

    # Predictions on training set
    Y_pred = model.predict(X)

    # log
    if log : 
        print(f"predictions on {step} set...")
        print(Y_pred)
        print()

    return Y_pred

# Confusion matrix
def c_matrix(y, y_pred, log=True, step="") :
    """ confusion matrix
    """
    confusion_m = confusion_matrix(y, y_pred)

    if log : 
        print(f"confusion matrix on {step} set : ")
        print(confusion_m)

    return confusion_m