#Goal: to define the randomly selected training genes and GO scores for predicting synapse genes

import pandas as pd
import numpy as np
import csv
import random

import ddot
from ddot import Ontology

from load_data_functions import get_gene_names


#functions for finding training genes==============================================================================
def random_select(genelist, big_pool, GO_genes, no_select):
	overlap=list(set(genelist)&set(big_pool)&set(GO_genes))
	print ('overlap', len(overlap))
	genes=sorted(overlap)
	random.seed(0)
	sel=random.sample(genes, no_select)
	print ('final', len(sel))
	return sel

#find the pos and neg training genes:
def find_pos_neg_input(syngo, big_pool, GO_genes):
	#syngo=get_gene_names(syngo_file) 
	#big_pool=get_gene_names(index_file)
	pos=random_select(syngo, big_pool, GO_genes, int(len(syngo)/2))
	#pos_df=pd.DataFrame({'genes': pos})
	#pos_df.to_csv('%s_positives.csv'%name)
	
	negatives=list(set(big_pool)-set(syngo))
	neg=random_select(negatives, big_pool, GO_genes, len(pos))
	#neg_df=pd.DataFrame({'genes': neg})
	#neg_df.to_csv('%s_negatives.csv'%name)
	return pos, neg

def make_genes_csv(genelist, first_word, df_name):
	df=pd.DataFrame({'genes': genelist})
	df.to_csv('%s_%s.csv'%(first_word, df_name))
	return df

#divide list of genes into five chunks for 5-fold cross-validation
def divide_5fold(genelist):
	chunk_size=int(len(genelist)/5)
	chunks=[genelist[i * chunk_size:(i+1) * chunk_size] for i in range((len(genelist)+chunk_size-1) // chunk_size )]
	return chunks

def find_pos_neg_chunks(positives, negatives):
	pos_chunks=divide_5fold(positives)
	neg_chunks=divide_5fold(negatives)
	return pos_chunks, neg_chunks

def define_training_test(positives, pos_chunks, negatives, neg_chunks, chunk_no):
	test_pos=pos_chunks[chunk_no]
	training_pos=list(set(positives)-set(test_pos))

	test_neg=neg_chunks[chunk_no]
	training_neg=list(set(negatives)-set(test_neg))

	training=training_pos+training_neg
	training.sort()
	#training_df=pd.DataFrame(training, columns=['genes'])
	#training_df.to_csv('%s_training_genes_%s.csv'%(name,chunk_no))
	test=test_pos+test_neg
	test.sort()
	#test_df=pd.DataFrame(test, columns=['genes'])
	#test_df.to_csv('%s_test_genes_%s.csv'%(name, chunk_no))
	print ('overlap', len(set(training)&set(test)))
	return training, test

def find_training_pos_neg(syngo, big_pool, GO_genes):
	pos, neg=find_pos_neg_input(syngo, big_pool, GO_genes)
	print ('pos', len(pos), 'neg', len(neg))
	all_training=list(set(pos+neg))
	all_training=sorted(all_training)
	print(len(all_training))
	return pos, neg, all_training
	
#find the pos and neg training genes:
def define_pos_neg_training(syngo, big_pool, GO_genes):
	pos, neg, all_training=find_training_pos_neg(syngo, big_pool, GO_genes)
	pos_df=make_genes_csv(pos, 'updated', 'positives')
	neg_df=make_genes_csv(neg, 'updated', 'negatives')
	return pos, neg, all_training

def find_pos_genes_in_training(training_genes, positives):
	overlap=list(set(training_genes)&set(positives))
	pos_training=list(set(overlap))
	#print (len(input_genes))
	return pos_training

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
	matrix.to_csv('%s_GO_training_score_matrix_for_big_pool_genes.csv'%name)
	return matrix

def define_GO_score_matrix(pos, neg, GO_human):
	GO_score_matrix=find_input_gene_GO_scores(pos, neg, GO_human)
	make_mat_csv(GO_score_matrix, 'syngo_GO_training_score_matrix_for_big_pool_genes.csv')
	return GO_score_matrix

def load_GO_score_matrix(go_mat_filename):
	df=pd.read_csv(go_mat_filename, index_col=[0])
	return df


