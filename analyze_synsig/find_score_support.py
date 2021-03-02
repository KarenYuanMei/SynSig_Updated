#Goal: find the relationship between the predicted scores and the number of supporting sources


import pandas as pd
import numpy as np
import csv

from scipy import stats

from scipy.stats import percentileofscore

import sys
#sys.path.append('../ML_functions/')
#import ROC_functions

#sys.path.append('../read_data_functions/')
#import load_data_functions

sys.path.append('../graph_functions/')
#import find_training_genes_functions
#import find_GO_scores
import graph_functions

#table=pd.read_csv('../run_ML/update_web_table.csv')
table=pd.read_csv('update_web_table.csv')

all_syn=table[table['Lit Sum']==3]
print (all_syn)

thresholds=np.arange(10, 110, 10)
print (thresholds)

totals=[]
syns=[]
syn_ratio=[]
for item in thresholds:
	perc=table['Synapse Percentile'].tolist()
	new=table[table['Synapse Percentile']<=item]
	total=new.shape[0]
	print (total)
	totals.append(total)
	syn_new=new[new['Lit Sum']==3]
	#print (syn_new)
	syn=syn_new.shape[0]
	print (syn)
	syns.append(syn)
	ratio=float(syn/(98)*100)
	syn_ratio.append(ratio)
	table = table[~table.index.isin(new.index)]
	#print (table)

print (totals)
print (syns)
print (syn_ratio)

labels=['<10', '10<x<20', '20<x<30', '30<x<40', '40<x<50', '50<x<60', '60<x<70', '70<x<80', '80<x<90', '90<x<100' ]
graph_functions.plot_bargraph(labels, syn_ratio, 'Pred Synapse Score Percentile', 'Percentage of Recovered Synapse Genes', 'syn_score')
