#Goal: to add description to training synapse gene names and non-synapse genes
#gene_name_description.txt was downloaded on April 9th from Biomart

#to annotate each SynSig protein to a molecular function category

import pandas as pd

import csv

import numpy as np

import ddot
from ddot import Ontology

import sys
sys.path.append('../ML_functions/')
import find_GO_scores

from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import sys
sys.path.append('../read_data_functions/')
import load_data_functions

sys.path.append('../ML_functions/')
import find_training_genes_functions
import find_GO_scores

import update_synsig_gene_func
import make_pred_db_data_table

def format_training_ms_count_df(pos_df, all_gl, all_gl_names, name):
	pos_genes=pos_df['genes'].tolist()
	all_counts=[]
	for item in all_gl:
		gl_count=make_pred_db_data_table.count_in_genelist(pos_genes, item)
		all_counts.append(gl_count)

	for i in range(len(all_counts)):
		pos_df[all_gl_names[i]]=all_counts[i]

	print (pos_df)
	pos_df['Lit Sum']=pos_df[['syngo', 'syndb', 'synsysnet']].sum(axis=1)
	
	pos_df['Exp Sum']=pos_df[['cortex', 'striatum', 'fetal', 'ngn2']].sum(axis=1)

	pos_df['All Sum']=pos_df[all_gl_names].sum(axis=1)

	# count_df['Synapse Percentile']=perc
	pos_df.to_csv('training_%s_ms_table.csv'%name)
	print (pos_df)
	return pos_df

def load_databases():
	syngo=load_data_functions.find_full_syngo()
	syndb=load_data_functions.find_full_SynDB()
	synsysnet=load_data_functions.find_full_synsysnet()

	cortex=load_data_functions.find_full_adult_cortex()
	striatum=load_data_functions.find_full_adult_striatum()
	fetal=load_data_functions.find_full_fetal()
	ngn2=load_data_functions.find_full_ngn2()
	return syngo, syndb, synsysnet, cortex, striatum, fetal, ngn2

def annotate_training_df(pos_genes, name):

	#add description all synsig genes:
	pos_desc=update_synsig_gene_func.add_desc(pos_genes)
	pos_desc.to_csv('training_%s_genes_desc.csv'%name)

	#add mf to all synsig genes:
	pos_desc_mf=update_synsig_gene_func.add_mf_function(pos_genes, pos_desc)
	pos_desc_mf.to_csv('training_%s_desc_mf.csv'%name)
	print (pos_desc_mf)

	syngo, syndb, synsysnet, cortex, striatum, fetal, ngn2=load_databases()

	all_gl=[syngo, syndb, synsysnet, cortex, striatum, fetal, ngn2]
	all_gl_names=['syngo', 'syndb', 'synsysnet', 'cortex', 'striatum', 'fetal', 'ngn2']

	format_training_ms_count_df(pos_desc_mf, all_gl, all_gl_names, name)
	

#find training pos genes:
# training_pos=pd.read_csv('../run_ML/ML_output/training_genes/updated_positives.csv')
# print (training_pos)
# pos_genes=training_pos['genes'].tolist()
# annotate_training_df(pos_genes, 'pos')

#find training negative genes:
training_neg=pd.read_csv('../run_ML/ML_output/training_genes/updated_negatives.csv')
print (training_neg)
neg_genes=training_neg['genes'].tolist()
annotate_training_df(neg_genes, 'neg')
annotate_training_df(neg_genes, 'neg')

