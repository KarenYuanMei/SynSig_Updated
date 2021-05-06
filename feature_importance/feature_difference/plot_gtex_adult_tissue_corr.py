#Goal: analyze number of transcripts in the human genes as reported by Ensembl to determine if there are differences between synapse and non-synapse genes in this feature

#source: Ensembl

import csv
import numpy as np
import math
import pandas as pd
from scipy import stats

from mlxtend.evaluate import permutation_test

import tkinter
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt

from scipy import stats

from numpy.random import seed 
from numpy.random import randn 
from scipy.stats import mannwhitneyu 

import ddot
from ddot import Ontology
import random

import seaborn as sns



import analyze_feature_diff

import sys
sys.path.append('../../graph_functions/')
import graph_functions

sys.path.append('../../read_data_functions/')
import load_data_functions

sys.path.append('../../ML_functions/')
import find_training_genes_functions

plt.style.use('seaborn-deep')
matplotlib.rcParams.update({'font.size': 22})


filename='gtex_rna_tissue_expression'
df=pd.read_csv('../../../../SynSig/features/normalized_%s.csv'%filename, index_col=[0])
df=df.drop(['Norm_Symbol'],axis=1)

print (df)

synsig_pos=load_data_functions.load_synsig()
synsig_neg=analyze_feature_diff.find_synapse_negative_pool(synsig_pos)
print (len(synsig_pos), len(synsig_neg))

ordered_pos_neg=synsig_pos+synsig_neg

ordered_df=df.loc[ordered_pos_neg]
ordered_df=ordered_df.T
print (ordered_df)


corr_df=ordered_df.corr()
print (corr_df)

#plt.imshow(corr_df, cmap='hot')
#plt.savefig('gtex_heatmap.pdf', format="pdf")
#plt.show()

sns.heatmap(corr_df)
plt.savefig('gtex_heatmap.pdf', format="pdf")