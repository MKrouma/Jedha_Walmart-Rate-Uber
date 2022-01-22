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