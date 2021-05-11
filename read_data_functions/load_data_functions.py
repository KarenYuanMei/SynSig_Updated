#Goal: to load different files into pandas

import pandas as pd
import pathlib

import os
import os.path

# import sys
# sys.path.append('../rwalk_functions/')
# import make_network_graph_functions

def get_gene_names(filename):
	df=pd.read_csv(filename)
	genes=df['genes'].tolist()
	return genes

def load_big_pool():
	p = pathlib.Path(__file__).resolve().parents[1]
	p = str(p)
	index_file= p + '/source_data_files/gene_pools/big_pool_genes_index.csv'
	big_pool=get_gene_names(index_file)
	return big_pool

#load synapse DB gene lists=======================================

# def find_syngo(big_pool, go_genes):
# 	syngo_file='../correct_db/corr_syngo_cc.csv'
# 	syngo=get_gene_names(syngo_file)
# 	syngo=list(set(syngo)&set(big_pool)&set(go_genes))
# 	return syngo

def find_syngo(big_pool, go_genes):
	p = pathlib.Path(__file__).resolve().parents[1]
	p = str(p)
	file= p + '/source_data_files/correct_db/corr_syngo_cc.csv'
	genes=get_gene_names(file)
	syngo=list(set(genes)&(set(big_pool)&set(go_genes)))
	return syngo

def find_full_syngo():
	p = pathlib.Path(__file__).resolve().parents[1]
	p = str(p)
	file= p + '/source_data_files/correct_db/corr_syngo_cc.csv'
	genes=get_gene_names(file)
	return genes



# def find_SynDB(big_pool):
# 	synDB_file='../correct_db/SynDB.csv'
# 	syndb=get_gene_names(synDB_file)
# 	syndb=list(set(syndb)&set(big_pool))
# 	return syndb

def find_SynDB(big_pool):
	p = pathlib.Path(__file__).resolve().parents[1]
	p = str(p)
	file= p + '/source_data_files/correct_db/SynDB.csv'
	genes=get_gene_names(file)
	syndb=list(set(genes)&set(big_pool))
	return syndb

# def find_synsysnet(big_pool):
# 	synsysnet_file='../correct_db/synsysnet.csv'
# 	synsysnet=get_gene_names(synsysnet_file)
# 	synsysnet=list(set(synsysnet)&set(big_pool))
# 	return synsysnet

def find_synsysnet(big_pool):
	p = pathlib.Path(__file__).resolve().parents[1]
	p = str(p)
	file= p + '/source_data_files/correct_db/synsysnet.csv'
	genes=get_gene_names(file)
	synsysnet=list(set(genes)&set(big_pool))
	return synsysnet


def load_synsig():
	p = pathlib.Path(__file__).resolve().parents[1]
	p = str(p)
	index_file= p + '/analyze_synsig/synsig_only.csv'
	synsig=get_gene_names(index_file)
	return synsig

def load_predicted_synsig_df():
	p = pathlib.Path(__file__).resolve().parents[1]
	p = str(p)
	index_file= p + '/run_ML/ML_output/new_brain_RNA_big_pool_novel_synapse_genes_avg_scores.csv'
	df=pd.read_csv(index_file, index_col=[0])
	return df

def load_predicted_ensig_df():
	p = pathlib.Path(__file__).resolve().parents[1]
	p = str(p)
	index_file= p + '/run_ML/ML_output/nb_brain_RNA_big_pool_novel_synapse_genes_avg_scores.csv'
	df=pd.read_csv(index_file, index_col=[0])
	return df


#load synapse mass spec screen lists====================================

def find_adult_cortex(big_pool):
	ctx_file='../../../SynSig/experimental_validation/weijun_ctx_uniprot.csv'
	ctx_genes=pd.read_csv(ctx_file, sep='\t')
	ctx_genes=ctx_genes['To'].tolist()
	ctx_genes=[x.upper() for x in ctx_genes]
	#print (ctx_genes)
	overlap_ctx=list(set(ctx_genes)&set(big_pool))
	return overlap_ctx

def find_adult_striatum(big_pool):
	str_file='../../../SynSig/experimental_validation/weijun_str_uniprot.csv'
	str_genes=pd.read_csv(str_file, sep='\t')
	str_genes=str_genes['To'].tolist()
	str_genes=[x.upper() for x in str_genes]
	#print (ctx_genes)
	overlap_str=list(set(str_genes)&set(big_pool))
	return overlap_str

def find_fetal(big_pool):
	fetal_file='../../../SynSig/experimental_validation/coba_fetal_brain.csv'
	fetal_genes=pd.read_csv(fetal_file)
	fetal_genes=fetal_genes['Norm_Symbol'].tolist()
	fetal_genes=[x.upper() for x in fetal_genes]
	#print (ctx_genes)
	overlap_fetal=list(set(fetal_genes)&set(big_pool))
	return overlap_fetal

def find_ngn2(big_pool):
	ngn2_file='../../../SynSig/experimental_validation/Coba_NGN2.csv'
	ngn2_genes=pd.read_csv(ngn2_file)
	ngn2_genes=ngn2_genes['Norm_Symbol'].tolist()
	ngn2_genes=[x.upper() for x in ngn2_genes]
	#print (ctx_genes)
	overlap_ngn2=list(set(ngn2_genes)&set(big_pool))
	return overlap_ngn2

#load hek293 genes==========================================================

def load_hek_genes():
	p = pathlib.Path(__file__).resolve().parents[1]
	p = str(p)
	#print (p)
	filename=p +'/source_data_files/expression_files/hek_genes.csv'
	#print (filename)
	genes=get_gene_names(filename)
	return genes


#load ppi df files=========================================================
def load_bioplex_file():
	p = pathlib.Path(__file__).resolve().parents[1]
	p = str(p)
	#print (p)
	filename=p +'/source_data_files/ppi_files/BioPlex 3 - HEK293T default edge.csv'
	#print (filename)
	return filename

def load_mentha_file():
	p = pathlib.Path(__file__).resolve().parents[1]
	p = str(p)
	#print (p)
	filename=p +'/source_data_files/ppi_files/Human_Mentha_converted.csv'
	#print (filename)
	return filename

