#Goal: to define the randomly selected training scores for predicting synapse genes

import pandas as pd
import numpy as np
import csv
import random

import ddot
from ddot import Ontology

import sys
sys.path.append('../read_data_functions/')
from load_data_functions import get_gene_names

#find GO scores for the training gene sets===================================================================
def find_GO_ont():
	ndex_server ='http://public.ndexbio.org' 
	ndex_user, ndex_pass = 'ym2', 'Synapse'
	go_human = Ontology.from_ndex('http://public.ndexbio.org/v2/network/bab8b805-2eb8-11eb-9e72-0ac135e8bacf')
	print (go_human)
	return go_human

def find_GO_score_matrix(go_human):
	#go_human=find_GO_ont()
	sim, genes=go_human.flatten()
	sim_df = pd.DataFrame(sim, index=genes, columns=genes)
	return sim_df

def find_input_gene_GO_scores(positive_genes, negative_genes, go_human):
	input_gene_names=list(set(positive_genes+negative_genes))
	print ('done1')
	GO_sim=find_GO_score_matrix(go_human)
	GO_genes=list(GO_sim.index)
	overlap=list(set(GO_genes)&set(input_gene_names))
	print ('done2')
	GO_score_matrix=GO_sim.loc[overlap, overlap]
	print ('done3')
	return GO_score_matrix

def make_mat_csv(mat, name):
	matrix.to_csv('../run_ML/ML_output/training_genes/%s_GO_training_score_matrix_for_big_pool_genes.csv'%name)
	return matrix

def define_GO_score_matrix(pos, neg, GO_human, name):
	GO_score_matrix=find_input_gene_GO_scores(pos, neg, GO_human)
	make_mat_csv(GO_score_matrix, name)
	return GO_score_matrix

def load_GO_score_matrix(go_mat_filename):
	df=pd.read_csv(go_mat_filename, index_col=[0])
	return df

