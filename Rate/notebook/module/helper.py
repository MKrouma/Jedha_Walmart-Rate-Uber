""" helper functions for project.
"""

# import
import os
import json 
import pickle
import pandas as pd


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