#Goal: find the relationship between the predicted scores and the number of supporting sources


import pandas as pd
import numpy as np
import csv

from scipy import stats

from scipy.stats import percentileofscore
from scipy.stats import hypergeom

import sys
sys.path.append('../graph_functions/')
import graph_functions

def find_syn_ratio(table):
	thresholds=np.arange(10, 110, 10)
	syn_ratio=[]
	for item in thresholds:
		perc=table['Synapse Percentile'].tolist()
		new=table[table['Synapse Percentile']<=item]
		total=new.shape[0]
		print (total)
		#totals.append(total)
		syn_new=new[new['Lit Sum']==3]
		#print (syn_new)
		syn=syn_new.shape[0]
		print (syn)
		#syns.append(syn)
		ratio=float(syn/(98)*100)
		syn_ratio.append(ratio)
		table = table[~table.index.isin(new.index)]
		#print (table)
	return syn_ratio

def plot_syn_ratio_scores():
	table=pd.read_csv('update_web_table.csv')

	all_syn=table[table['Lit Sum']==3]
	all_syn=all_syn.shape[0]

	syn_ratio=find_syn_ratio(table)

	labels=['<10', '10<x<20', '20<x<30', '30<x<40', '40<x<50', '50<x<60', '60<x<70', '70<x<80', '80<x<90', '90<x<100' ]
	graph_functions.plot_bargraph(labels, syn_ratio, 'Pred Synapse Score Percentile', 'Percentage of Recovered Synapse Genes', 'syn_score')


def find_hypergeometric(genelist1, genelist2, M):
	overlap=list(set(genelist1)&set(genelist2))
	#M=20000
	N=len(genelist1)
	n=len(genelist2)
	x=len(overlap)
	pval = hypergeom.sf(x-1, M, n, N)

	rv = hypergeom(M, n, N)
	distr = np.arange(0, n+1)
	print (N, n, x)
	prob = rv.pmf(distr)

	maximum=np.max(prob)
	result = np.where(prob == maximum)
	#print (result)
	#result=result.tolist()
	result=result[0]
	#print (result)
	fold=x/result
	fold=fold.tolist()
	#print ('Fold Enrichment', fold)
	#print ('hypergeometric p-value', pval)
	return fold, pval


def find_synsig_all_support():
	table=pd.read_csv('../run_ML/update_web_table.csv')

	synsig=table[table['avg_scores']>4.45]
	synsig=synsig['genes'].tolist()

	M=table['genes'].tolist()
	M=len(M)

	source_no=np.arange(0,8,1)
	print (source_no)

	folds=[]
	pvals=[]
	for item in source_no:
		source=table[table['All Sum']==item]
		source_genes=source['genes'].tolist()
		#print (source)
		#overlap=list(set(synsig)&set(source_genes))
		fold, pval=find_hypergeometric(synsig, source_genes, M)
		folds.append(fold)
		pvals.append(pval)

	print (folds)
	print(pvals)
	return folds, pvals



if __name__ == '__main__':
	#find_synsig_all_support()
	#plot is done in R

	table=pd.read_csv('../run_ML/update_web_table.csv')

	synsig_df=table[table['avg_scores']>4.45]
	synsig=synsig_df['genes'].tolist()

	dbs=np.arange(0,4)
	lit_no=[]
	lit_ratio=[]
	for item in dbs:
		lit=synsig_df[synsig_df['Lit Sum']==item]
		lit_genes=lit['genes'].tolist()
		lit_no.append(len(lit_genes))
		lit_ratio.append(float(len(lit_genes)/len(synsig)*100))

	print (lit_no)
	print (lit_ratio)

	labels=['New', '1', '2', '3']

	graph_functions.plot_bargraph(labels, lit_no, 'No of Prev. Supporting DBs', 'No of SynSig', 'synsig_new_genes_bar')




