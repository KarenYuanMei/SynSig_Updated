#Goal: find the relationship between the predicted scores and the number of supporting sources


import pandas as pd
import numpy as np
import csv

from scipy.stats import percentileofscore

import sys
sys.path.append('../ML_functions/')
import ROC_functions

sys.path.append('../read_data_functions/')
import load_data_functions

sys.path.append('../ML_functions/')
import find_training_genes_functions
import find_GO_scores

df=pd.read_csv('update_web_table.csv')
scores=df['avg_scores'].tolist()
supp=df['All Sum'].tolist()

all_means=[]
all_sems=[]
for i in range(8):
	df=df[df['All Sum']==i]
	scores=df['avg_scores'].tolist()
	mean=np.mean(scores)
	all_means.append(mean)
	sem=stat.sem(scores)
	all_sems.append(sem)

print (all_means)
print (all_sems)
