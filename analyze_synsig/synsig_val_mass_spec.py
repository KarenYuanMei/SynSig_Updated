#Goal: to validate Synsig predictions with mouse adult mass spectrometry screen results; distinguish between all SynSig vs. new SynSig

import pandas as pd
import numpy as np
import csv

from scipy.stats import hypergeom

import sys
sys.path.append('../read_data_functions/')
import load_data_functions

sys.path.append('../ML_functions/')
import find_training_genes_functions
import find_GO_scores

def find_hypergeometric(genelist1, genelist2, big_pool, training):
	genes1=list(set(genelist1)-set(training))
	genes2=list(set(genelist2)-set(training))
	pool_no_training=list(set(big_pool)-set(training))
	overlap=list(set(genes1)&set(genes2))
	M=len(pool_no_training)
	#print (M)
	#M=20000
	N=len(genes1)
	n=len(genes2)
	x=len(overlap)
	pval = hypergeom.sf(x-1, M, n, N)

	rv = hypergeom(M, n, N)
	distr = np.arange(0, n+1)
	#print (N, n, x)
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

def find_adult_genes(big_pool):

	cortex=load_data_functions.find_adult_cortex(big_pool)
	#print (len(cortex))

	striatum=load_data_functions.find_adult_striatum(big_pool)
	#print (len(striatum))

	ctx_str=list(set(cortex)&set(striatum))
	return cortex, striatum, ctx_str

def find_fetal_genes(big_pool):

	ngn2=load_data_functions.find_ngn2(big_pool)

	fetal=load_data_functions.find_fetal(big_pool)

	ngn2_fetal=list(set(ngn2)&set(fetal))

	return ngn2, fetal, ngn2_fetal

def find_screen_enrichment(genelist, screen_list, screen_labels, big_pool, training, name, synapse_status):

	folds=[]
	pvals=[]
	for item in screen_list:
		fold, pval=find_hypergeometric(genelist, item, big_pool, training)
		folds.append(fold[0])
		pvals.append(pval)

	df=pd.DataFrame({'tissues': screen_labels, 'enrichment': folds, 'significance': pvals})
	df['age']=synapse_status
	#df.to_csv('%s_mass_spec.csv'%name)
	return df

def load_syngo(big_pool):
	human_ont=find_GO_scores.find_GO_ont()
	go_genes=human_ont.genes
	syngo=load_data_functions.find_syngo(big_pool, go_genes)
	return syngo


if __name__ == '__main__':

	synsig=load_data_functions.load_synsig()

	big_pool=load_data_functions.load_big_pool()

	syngo=load_syngo(big_pool)
	
	training=find_training_genes_functions.load_pos_neg_training()

	cortex, striatum, ctx_str=find_adult_genes(big_pool)

	adult_df=find_screen_enrichment(synsig, [cortex, striatum, ctx_str], ['cortex', 'striatum', 'striatum/ctx'], big_pool, training, 'adult', 'all' )

	new=list(set(synsig)-set(syngo))
	new_adult_df=find_screen_enrichment(new, [cortex, striatum, ctx_str], ['cortex', 'striatum', 'striatum/ctx'], big_pool, training, 'new_adult', 'new')

	adult=pd.concat([adult_df, new_adult_df])
	print (adult)
	adult.to_csv('adult_mass_spec.csv')

	ngn2, fetal, ngn2_fetal=find_fetal_genes(big_pool)

	fetal_df=find_screen_enrichment(synsig, [ngn2, fetal, ngn2_fetal], ['a ngn2', 'fetal', 'ngn2/fetal'], big_pool, training, 'fetal', 'all')

	new_fetal_df=find_screen_enrichment(new, [ngn2, fetal, ngn2_fetal], ['a ngn2', 'fetal', 'ngn2/fetal'], big_pool, training, 'new_fetal', 'new')

	fetal=pd.concat([fetal_df, new_fetal_df])
	print (fetal)
	fetal.to_csv('fetal_mass_spec.csv')

	#plot the bargraphs using R
	