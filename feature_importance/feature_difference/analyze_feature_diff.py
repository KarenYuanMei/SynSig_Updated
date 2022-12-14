#Goal: analyze number of transcripts in the human genes as reported by Ensembl to determine if there are differences between synapse and non-synapse genes in this feature

#source: Ensembl

#Fig 4A

import csv
import numpy as np
import math
import pandas as pd
from scipy import stats

import statistics

from mlxtend.evaluate import permutation_test

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

import sys
sys.path.append('../../graph_functions/')
import graph_functions

sys.path.append('../../read_data_functions/')
import load_data_functions

sys.path.append('../../ML_functions/')
import find_training_genes_functions

plt.style.use('seaborn-deep')
matplotlib.rcParams.update({'font.size': 16})

def find_permutation(positives, negatives):
	p_value=permutation_test(positives, negatives, method='approximate', num_rounds=10000, seed=0)
	#print (p_value)
	return p_value

def find_students_test(positives, negatives):
	tstat, pvalue=stats.ttest_ind(positives, negatives, equal_var = False)
	return tstat, pvalue


#find genes that are not predicted to be in SynSig, but are in the big_pool
def find_synapse_negative_pool(synsig):
	big_pool=load_data_functions.load_big_pool()
	all_training=find_training_genes_functions.load_pos_neg_training()
	no_train_pool=list(set(big_pool)-set(all_training))
	negative_pool=list(set(no_train_pool)-set(synsig))
	return negative_pool

def find_gene_feature_values(df, genelist):
	df=df.set_index('Norm_Symbol')
	focused_df=df.loc[genelist]
	print (focused_df)
	focused_values=focused_df.iloc[:,-1].tolist()
	#print (focused_values)
	#print (focused_values[:5])
	return focused_values


if __name__ == '__main__':
	
	features=['ENSEMBL_aa_length', "cds_length", "exon_no", "gc_content", "trans_len", "gene_length", "Ensembl_isoform_no", "trans_count", "pFAM_domain_number", "Phosphosite_hu_no", "protein_mass"]
	renamed=['Amino Acid Length', "Coding Sequence Length", 'Exon Number', 'GC Content', "Transcript Length", 'Gene Length', 'Isoform Number', 'Transcript Count', 'Protein Domain Number', 'Phosphorylation Site Number', 'Protein Mass']

	folds=[]
	pvals=[]
	synsig_means=[]
	synsig_sems=[]
	negatives_means=[]
	negatives_sems=[]
	for item in features:
		length_features=['ENSEMBL_aa_length', "cds_length", "gene_length", "trans_len", "protein_mass"]
		filename='../../../../SynSig/features/normalized_%s.csv'%item

		data_type=item
		#print (item)

		df=pd.read_csv(filename)
		#print (df)

		synsig_pos=load_data_functions.load_synsig()
		#print (len(synsig))

		synsig_neg=find_synapse_negative_pool(synsig_pos)

		#print (len(synsig_neg))
		if item in length_features:

			syn_val=find_gene_feature_values(df, synsig_pos)
			syn_val = [x / 1000 for x in syn_val]
			#print (syn_val[:5])
			neg_val=find_gene_feature_values(df, synsig_neg)
			neg_val = [x / 1000 for x in neg_val]

		else:
			syn_val=find_gene_feature_values(df, synsig_pos)
			neg_val=find_gene_feature_values(df, synsig_neg)
		#print (neg_val[:5])

		#p_value=find_permutation(syn_val, neg_val)
		#print (p_value)
		synsig_mean=statistics.mean(syn_val)
		synsig_means.append(synsig_mean)

		synsig_sem=stats.sem(syn_val)
		synsig_sems.append(synsig_sem)

		neg_mean=statistics.mean(neg_val)
		negatives_means.append(neg_mean)

		neg_sem=stats.sem(neg_val)
		negatives_sems.append(neg_sem)


		fold=statistics.mean(syn_val)/statistics.mean(neg_val)
		print (fold)

		tstat, pvalue=find_students_test(syn_val, neg_val)

		print (tstat, pvalue)

		folds.append(fold)
		pvals.append(pvalue)

	df=pd.DataFrame({'Feature': renamed, 'SynSig Mean': synsig_means, 'Negative Mean': negatives_means, 'SynSig sem': synsig_sems, 'Negative Sem': negatives_sems, 'Fold': folds, 'PVals': pvals})
	print (df)
	df.to_csv('synsig_feature_diff.csv')
	# 	tstat, pvalue=find_students_test(syn_val, neg_val)
	# 	print (tstat, pvalue)

	def plot_bargraph(labels, mean_values, xlabel, ylabel, name):
		x_pos=np.arange(len(labels))
		#plt.bar(labels, mean_values, yerr=sem, color=['#7f6d5f', '#2d7f5e', '#557f2d','silver', 'dimgray', 'rosybrown'], align='center', ecolor='black', capsize=10)
		plt.bar(labels, mean_values, align='center', color='#2d7f5e', ecolor='black', capsize=10)

		#plt.ylim(1, 10**5)
		#plt.ylim(0.5, 1)
		#plt.yscale('log')
		# Create legend & Show graphic
		#plt.legend()
		#y_ticks = np.arange(0, 25, 5)
		#plt.yticks(y_ticks)
		plt.xlabel(xlabel, fontweight='bold')
		plt.ylabel(ylabel, fontweight='bold')
		#plt.xticks(rotation=45)
		plt.savefig(name+'.svg', format="svg")
		plt.close()


	plot_bargraph(features, folds, 'Features', 'Fold over Negatives', 'feature_diff')

	ax=df[['SynSig Mean', 'Negative Mean']].plot(kind='bar', yerr=df[['SynSig sem', 'Negative Sem']].values.T, color=['red', 'gray'], alpha = 0.5,error_kw=dict(ecolor='k'), bottom=1)
	xlabel=features
	ax.set_xticklabels(renamed)
	ax.set_yscale('log')
	#plt.xticks(xlabel, rotation = '90')
	#plt.xlabel(xlabel, fontweight='bold')
	plt.savefig('feature_difference.svg', format="svg", bbox_inches='tight')
		
