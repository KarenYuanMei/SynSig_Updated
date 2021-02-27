#Goal: to load different files into pandas

import pandas as pd

def get_gene_names(filename):
	df=pd.read_csv(filename)
	genes=df['genes'].tolist()
	return genes

def find_syngo(big_pool, go_genes):
	syngo_file='../correct_db/corr_syngo_cc.csv'
	syngo=get_gene_names(syngo_file)
	syngo=list(set(syngo)&set(big_pool)&set(go_genes))
	return syngo

def find_SynDB(big_pool):
	synDB_file='../correct_db/SynDB.csv'
	syndb=get_gene_names(synDB_file)
	syndb=list(set(syndb)&set(big_pool))
	return syndb

def find_synsysnet(big_pool):
	synsysnet_file='../correct_db/synsysnet.csv'
	synsysnet=get_gene_names(synsysnet_file)
	synsysnet=list(set(synsysnet)&set(big_pool))
	return synsysnet

