#Goal: 1) make table of pred scores, db, and mass spec data

import pandas as pd
import numpy as np
import csv

import sys
sys.path.append('../ML_functions/')
import ROC_functions

sys.path.append('../read_data_functions/')
import load_data_functions

sys.path.append('../ML_functions/')
import find_training_genes_functions
import find_GO_scores

pred=ROC_functions.load_predicted_df()
print (pred)
pred_genes=pred['genes'].tolist()
pred_scores=pred['avg_scores'].tolist()

big_pool=find_training_genes_functions.load_big_pool()

human_ont=find_GO_scores.find_GO_ont()
go_genes=human_ont.genes

syngo=load_data_functions.find_syngo(big_pool, go_genes)
syndb=load_data_functions.find_SynDB(big_pool)
synsysnet=load_data_functions.find_synsysnet(big_pool)

def count_in_genelist(pred_genes, genelist):
	count=[]
	for gene in pred_genes:
		if gene in genelist:
			entry=1
		else:
			entry=0
		count.append(entry)
	return count

syngo_count=count_in_genelist(pred_genes, syngo)
syndb_count=count_in_genelist(pred_genes, syndb)
synsysnet_count=count_in_genelist(pred_genes, synsysnet)

count_df=pd.DataFrame({'genes': pred_genes, 'synapse_sim_score': pred_scores, SynGO': syngo_count, 'SynDB': syndb_count, 'SynSysNet': synsysnet_count'})
print (count_df)

count_df=count_df.sort_values(by='synapse_sim_score')
print (count_df)