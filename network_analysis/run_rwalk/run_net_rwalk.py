#Goal: use the environment jh_network to run random walk

import scipy as sp
import networkx as nx
import pandas as pd

import numpy as np
#import matplotlib.pyplot as plt
import sys
sys.path.append('../rwalk_functions/')
import make_network_graph_functions

sys.path.append('../../read_data_functions')
import load_data_functions


sys.path.append('../ppi_files/')

filename='../ppi_files/Human_Mentha_converted.csv'
df=make_network_graph_functions.make_mentha_df(filename)
#print (df)

G=make_network_graph_functions.make_network_G(df)
print (G.number_of_edges())
nodes=list(G.nodes())

pos=load_data_functions.get_gene_names('../../run_ML/ML_output/training_genes/updated_positives.csv')
print (len(pos))

seeds=list(set(pos)&set(nodes))
print (len(seeds))

def find_nodesets(G, seeds):
	no_per_fold=len(seeds)/5
	keys=['first', 'second', 'third', 'fourth', 'fifth']
	seed_lists = [seeds[x:x+no_per_fold] for x in range(0, len(seeds), no_per_fold)]
	val_lists=[]
	for item in seed_lists:
		val_list=list(set(seeds)-set(item))
		val_lists.append(val_list)
	ordered=[]
	for i in range(len(seed_lists)):
		each=seed_lists[i]+val_lists[i]
		ordered.append(each)
	nodesets=dict(zip(keys, ordered))
	return nodesets

nodesets=find_nodesets(G, seeds)
#print (nodesets)

neg=load_data_functions.get_gene_names('../../run_ML/ML_output/training_genes/updated_negatives.csv')
print (len(neg))
neg=list(set(neg)&set(nodes))
print (len(neg))