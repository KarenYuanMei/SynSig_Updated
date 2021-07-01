#Goal: 1) make table of pred scores, db, and mass spec data

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


def load_synapse_lists(big_pool, go_genes):
	syngo=load_data_functions.find_syngo(big_pool, go_genes)
	syndb=load_data_functions.find_SynDB(big_pool)
	synsysnet=load_data_functions.find_synsysnet(big_pool)

	cortex=load_data_functions.find_adult_cortex(big_pool)
	striatum=load_data_functions.find_adult_striatum(big_pool)
	fetal=load_data_functions.find_fetal(big_pool)
	ngn2=load_data_functions.find_ngn2(big_pool)
	return syngo, syndb, synsysnet, cortex, striatum, fetal, ngn2

def load_full_databases():
	syngo=load_data_functions.find_full_syngo()
	syndb=load_data_functions.find_full_SynDB()
	synsysnet=load_data_functions.find_full_synsysnet()

	cortex=load_data_functions.find_full_adult_cortex()
	striatum=load_data_functions.find_full_adult_striatum()
	fetal=load_data_functions.find_full_fetal()
	ngn2=load_data_functions.find_full_ngn2()
	return syngo, syndb, synsysnet, cortex, striatum, fetal, ngn2

def count_in_genelist(pred_genes, genelist):
	count=[]
	for gene in pred_genes:
		if gene in genelist:
			entry=1
		else:
			entry=0
		count.append(entry)
	return count

def find_synapse_stat(count_df):
	synsig=count_df['SynSig'].tolist()
	lit_sum=count_df['Lit Sum'].tolist()

	status=[]
	for i in range(len(synsig)):
		if synsig[i]=='no':
			entry='no'
		else:
			if lit_sum[i]==0:
				entry='new'
			else:
				entry='old'
		status.append(entry)

	count_df['Synapse_Status']=status
	print (count_df)
	return count_df

def find_synapse_perc(count_df):
	count_df=pred.sort_values(by='avg_scores', ascending=False)
	scores=count_df['avg_scores'].tolist()

	perc=[]
	for item in scores:
		percentile = percentileofscore(scores, item)
		perc.append(percentile)
	return perc

def format_count_df(pred, all_gl, all_gl_names):
	pred_genes=pred['genes'].tolist()
	all_counts=[]
	for item in all_gl:
		gl_count=count_in_genelist(pred_genes, item)
		all_counts.append(gl_count)

	for i in range(len(all_counts)):
		pred[all_gl_names[i]]=all_counts[i]

	count_df=pred.sort_values(by='avg_scores', ascending=False)

	count_df['Lit Sum']=count_df[['syngo', 'syndb', 'synsysnet']].sum(axis=1)
	
	count_df['Exp Sum']=count_df[['cortex', 'striatum', 'fetal', 'ngn2']].sum(axis=1)

	count_df['All Sum']=count_df[all_gl_names].sum(axis=1)

	count_df['SynSig'] = np.where(count_df['avg_scores']>4.45, 'yes', 'no')

	count_df=find_synapse_stat(count_df)

	perc=find_synapse_perc(count_df)
	count_df['Synapse Percentile']=perc
	#count_df.to_csv('update_web_table.csv')
	return count_df

def make_pred_db_table_with_bigpool():
	pred=ROC_functions.load_predicted_df()
	print (pred)

	big_pool=find_training_genes_functions.load_big_pool()

	human_ont=find_GO_scores.find_GO_ont()
	go_genes=human_ont.genes

	syngo, syndb, synsysnet, cortex, striatum, fetal, ngn2=load_synapse_lists(big_pool, go_genes)
	all_gl=[syngo, syndb, synsysnet, cortex, striatum, fetal, ngn2]
	all_gl_names=['syngo', 'syndb', 'synsysnet', 'cortex', 'striatum', 'fetal', 'ngn2']
	count_df=format_count_df(pred, all_gl, all_gl_names)
	count_df.to_csv('update_web_table.csv')

def make_pred_db_table_with_full_genes():
	pred=ROC_functions.load_predicted_df()
	print (pred)
	syngo, syndb, synsysnet, cortex, striatum, fetal, ngn2=load_full_databases()
	all_gl=[syngo, syndb, synsysnet, cortex, striatum, fetal, ngn2]
	all_gl_names=['syngo', 'syndb', 'synsysnet', 'cortex', 'striatum', 'fetal', 'ngn2']
	count_df=format_count_df(pred, all_gl, all_gl_names)
	count_df.to_csv('update_web_table_full_db.csv')



if __name__ == '__main__':
	make_pred_db_table_with_full_genes()

	

	