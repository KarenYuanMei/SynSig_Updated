#Goal: 1) make table of pred scores, db, and mass spec data

import pandas as pd
import numpy as np
import csv

import sys
sys.path.append('../ML_functions/')
import ROC_functions

pred=ROC_functions.load_predicted_df()
print (pred)