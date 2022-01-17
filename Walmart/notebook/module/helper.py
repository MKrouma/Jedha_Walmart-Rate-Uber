""" helper functions for project.
"""

# import
import os
import json 
import pickle
import pandas as pd


# load_pickle
# helper function : load_pickle_array
def load_pickle_array(file_name) :
    file = open(file_name, 'rb')

    # dump information to that file
    narray = pickle.load(file)

    # close the file
    file.close()

    return narray