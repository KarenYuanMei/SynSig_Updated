#Goal: analyze the pearson correlation between synsig to synsig, synsig to non-synsig to determine if there are differences between synapse and non-synapse genes in this feature

#source: Gtex

#Fig4B

import csv
import numpy as np
import math
import pandas as pd
from scipy import stats

from mlxtend.evaluate import permutation_test

import tkinter
import matplotlib
#matplotlib.use("TKAgg")
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

from statistics import mean

import sys
sys.path.append('../../graph_functions/')
import graph_functions

sys.path.append('../../read_data_functions/')
import load_data_functions

sys.path.append('../../ML_functions/')
import find_training_genes_functions

plt.style.use('seaborn-deep')
matplotlib.rcParams.update({'font.size': 22})
plt.figure(figsize=(8,6))



filename='gtex_rna_tissue_expression'
df=pd.read_csv('../../../../SynSig/features/normalized_%s.csv'%filename, index_col=[0])
df=df.drop(['Norm_Symbol'],axis=1)

print (df)

#big_pool=load_data_functions.load_big_pool()

synsig_pos=load_data_functions.load_synsig()
synsig_neg=analyze_feature_diff.find_synapse_negative_pool(synsig_pos)
#synsig_neg=random.sample(synsig_neg, len(synsig_pos))
print (len(synsig_pos), len(synsig_neg))

syngo_training_pos=find_training_genes_functions.load_pos_training()

ordered_pos_neg=synsig_pos+synsig_neg
print (ordered_pos_neg[:10])

ordered_df=df.loc[ordered_pos_neg]
print (ordered_df)

ordered_df=ordered_df.T

corr_df=ordered_df.corr()



corr_df['avg']=corr_df.mean(axis=1)
print (corr_df)

pos=corr_df.loc[synsig_pos]
pos_pos=pos[synsig_pos]
pos_pos['avg']=pos_pos.mean(axis=1)
pos_pos_val=pos_pos['avg'].tolist()
print (pos_pos_val)

pos_neg=pos[synsig_neg]
pos_neg['avg']=pos_neg.mean(axis=1)
pos_neg_val=pos_neg['avg'].tolist()


neg=corr_df.loc[synsig_neg]
neg_neg=neg[synsig_neg]
neg_neg['avg']=neg_neg.mean(axis=1)
neg_neg_val=neg_neg['avg'].tolist()

pos_pos_val = [x for x in pos_pos_val if str(x) != 'nan']
pos_neg_val = [x for x in pos_neg_val if str(x) != 'nan']
neg_neg_val = [x for x in neg_neg_val if str(x) != 'nan']


print (mean(pos_pos_val))
print (mean(pos_neg_val))
print (mean(neg_neg_val))

print (stats.sem(pos_pos_val))
print (stats.sem(pos_neg_val))
print (stats.sem(neg_neg_val))

# im=plt.imshow(final_corr_df, cmap="YlGnBu", vmin=-1, vmax=1)
# plt.colorbar(im)
# plt.savefig('gtex_heatmap.pdf', format="pdf")
#plt.show()
ttest, pval=analyze_feature_diff.find_students_test(pos_pos_val, pos_neg_val)
print (ttest, pval)

ttest, pval=analyze_feature_diff.find_students_test(neg_neg_val, pos_neg_val)
print (ttest, pval)
#sns.heatmap(corr_with_syngo_df)
#plt.savefig('gtex_heatmap.pdf', format="pdf")

# def plot_training_distributions(positives, negatives, col_title):

# 	bins=np.histogram(np.hstack((positives,negatives)), bins=40)[1] #get the bin edges
	
# 	plt.hist(positives, bins, alpha=0.5, edgecolor='black', linewidth=0.5, color='darkorange')
# 	plt.hist(negatives, bins, alpha=0.5, edgecolor='black', linewidth=0.5, color='gray')

# 	#plt.ylabel('Non-Synapse Genes in Brain')
# 	plt.xlabel('%s'%col_title, fontweight='bold')
# 	plt.ylabel('Frequency', fontweight = 'bold')
# 	#plt.xscale('log')
# 	#plt.yscale('log')
# 	plt.legend(labels=['SynSig to SynSig', 'SynSig to Non-SynSig'])
# 	#plt.show()
# 	#plt.close()
# 	plt.grid(b=False)
# 	plt.savefig('gtex_adult_tissue.svg', bbox_inches='tight')
# 	plt.show()
# 	plt.close()

def plot_training_distributions(positives, negatives, col_title):

	bins=np.histogram(np.hstack((positives,negatives)), bins=40)[1] #get the bin edges
	
	plt.hist(positives, bins, alpha=0.5, edgecolor='black', linewidth=0.5, color='darkorange', bottom=10**(-9), density=True)
	plt.hist(negatives, bins, alpha=0.5, edgecolor='black', linewidth=0.5, color='bisque', bottom=10**(-9),  density=True)

	#plt.ylabel('Non-Synapse Genes in Brain')
	plt.xlabel('%s'%col_title, fontweight='bold')
	plt.ylabel('Probability', fontweight = 'bold')
	#plt.xticks(np.arange(min(x), max(x)+1, 5000))
	#plt.xscale('log')
	plt.legend(labels=['SynSig to SynSig', 'SynSig to Non-SynSig'])
	
	#plt.ylim(0, 1)
	#plt.show()
	#plt.close()
	plt.grid(b=False)
	plt.savefig('gtex_adult_tissue.svg', bbox_inches='tight')
	plt.show()
	plt.close()

plot_training_distributions(pos_pos_val, pos_neg_val, 'correlations')

data = [pos_pos_val, pos_neg_val, neg_neg_val]
  
fig = plt.figure(figsize =(8, 6))
  
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
  
# Creating plot
bp = ax.boxplot(data)

#plt.savefig('gtex_adult_tissue.pdf', bbox_inches='tight')
