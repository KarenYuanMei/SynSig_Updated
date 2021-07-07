#Goal: find the relationship between the predicted scores and the number of supporting sources

import tkinter
import pandas as pd
import numpy as np
import csv

from scipy import stats

from scipy.stats import percentileofscore
from scipy.stats import hypergeom

import sys
sys.path.append('../graph_functions/')
import graph_functions

sys.path.append('../read_data_functions/')
import load_data_functions

sys.path.append('../ML_functions/')
import find_GO_scores

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def find_hypergeometric(genelist1, genelist2, M):
	overlap=list(set(genelist1)&set(genelist2))
	#M=20000
	N=len(genelist1)
	n=len(genelist2)
	x=len(overlap)

	print ('M', M, 'N', N, 'n', n, 'x', x)
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

def find_exp_support(genelist):
	table=pd.read_csv('../run_ML/update_web_table.csv')
	M=table['genes'].tolist()
	M=len(M)

	source_no=np.arange(0,5,1)
	print (source_no)

	folds=[]
	pvals=[]
	for item in source_no:
		screen=table[table['Exp Sum']==item]
		screen_genes=screen['genes'].tolist()
		#print (source)
		#overlap=list(set(synsig)&set(source_genes))
		fold, pval=find_hypergeometric(genelist, screen_genes, M)
		folds.append(fold[0])
		pvals.append(pval)

	#print (folds)
	#print(pvals)
	return folds, pvals

def make_genelist_enrich_df():
		dfs=[]
		for i in range(len(labels)):
			folds, pvals=find_exp_support(genelists[i])
			df=pd.DataFrame({'Genelists': labels[i], '%s_Enrichment'%labels[i]: folds, '%s_Significance'%labels[i]: pvals})
			print (df)
			dfs.append(df)
			

		final=pd.concat(dfs, axis=1)
		
		final.index.name = 'Found in No. of Mass Spec Screens'
		print (final)

		final.to_csv('support_screens.csv')

		final=final.reset_index()
		print (final)
		return final

def plot_enrich_df(final):
	plt.plot( 'Found in No. of Mass Spec Screens', 'synsig_Enrichment', data=final, marker='o', markerfacecolor='darkgreen', markersize=18, color='darkgreen', linewidth=1)
	plt.plot( 'Found in No. of Mass Spec Screens', 'syngo_Enrichment', data=final, marker='o', markerfacecolor='gray', markersize=18, color='gray', linewidth=1)
	plt.ylim([0, 5])
	#plt.xlim([0, 4])
	
	# show legend
	plt.legend()

	# show graph
	plt.show()


if __name__ == '__main__':
	#find_synsig_all_support()
	#plot is done in R

	big_pool=load_data_functions.load_big_pool()
	go_human=find_GO_scores.find_GO_ont()
	go_genes=go_human.genes

	synsig=load_data_functions.load_synsig()
	#find_exp_support(synsig)

	syngo=load_data_functions.find_syngo(big_pool, go_genes)
	#find_exp_support(syngo)

	synDB=load_data_functions.find_SynDB(big_pool)
	#find_exp_support(synDB)

	synsysnet=load_data_functions.find_synsysnet(big_pool)
	#find_exp_support(synsysnet)

	#new=list(set(synsig)-set(syngo))
	new=list(set(synsig)-set(syngo))

	folds, pvals=find_exp_support(new)
	print (folds)
	print (pvals)

	supp_sources=np.arange(0,5,1)

	plt.plot(supp_sources, folds, marker='o', markerfacecolor='darkred', markeredgecolor='gray', markersize=15, color='darkred', linewidth=3, label='Genes in SynSig and not in SynGO')
	plt.ylim([0, 4])
	#plt.xlim([0, 4])
	
	# show legend
	plt.legend()

	# show graph
	plt.show()

	# genelists=[synsig, syngo, synDB, synsysnet]
	# labels=['synsig', 'syngo', 'synDB', 'synsysnet']

	# final=make_genelist_enrich_df()

	# plot_enrich_df(final)

	

	# no_sources=np.arange(0,5,1)

	# dfs=[]
	# for i in range(len(labels)):
	# 	folds, pvals=find_exp_support(genelists[i])
	# 	#label_name=labels[i]*4

	# 	df=pd.DataFrame({'Genelists': labels[i], 'Enrichment': folds, 'Significance': pvals})
	# 	df['Supp_Sources']=no_sources

	# 	print (df)
	# 	dfs.append(df)

	# final=pd.concat(dfs)
	# print (final)
	# final.to_csv('support_screens.csv')

			

		