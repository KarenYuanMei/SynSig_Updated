#Goal: for each molecular function category, distinguish between known vs. new synapse genes


import pandas as pd

import csv

import numpy as np

import ddot
from ddot import Ontology

import sys
sys.path.append('../ML_functions/')
import find_GO_scores

sys.path.append('../read_data_functions/')
import load_data_functions

from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


#compile all of the function files into a single file for Synsig
def compile_all_synsig_func():
	file1='synsig_desc_function.csv'
	file2='synsig_mf_function.csv'
	file3='synsig_met_function.csv'
	file4='synsig_unknown_function.csv'

	first_two=[file1, file2]

	ann_gene_dfs=[]
	for item in first_two:
		df=pd.read_csv(item, index_col=[0])
		#print (df)
		annotated=df[df['Function Total']>0]
		#print (annotated)
		ann_gene_dfs.append(annotated)

	final=pd.concat(ann_gene_dfs)
	final.insert(loc=15, column='Other Enzymes', value=0)
	final.insert(loc=16, column='Other protein binders', value=0)
	final.insert(loc=17, column='unknown functions', value=0)
	print (final)

	cols=list(final.columns)

	col_no=np.arange(3,15,1).tolist()
	#print (col_no)

	df3=pd.read_csv(file3, index_col=[0])
	annotated3=df3[df3['Function Total']>0]

	for item in col_no:
		annotated3.insert(loc=item, column=cols[item], value=0)

	annotated3.insert(loc=16, column='Other protein binders', value=0)
	annotated3.insert(loc=17, column='unknown functions', value=0)
	print (annotated3)


	df4=pd.read_csv(file4, index_col=[0])
	annotated4=df4[df4['Function Total']>0]
	#print (annotated)
	for item in col_no:
		annotated4.insert(loc=item, column=cols[item], value=0)

	annotated4.insert(loc=15, column='Other Enzymes', value=0)
	print (annotated4)

	final=pd.concat([final, annotated3, annotated4])
	#print (final)
	final.to_csv('all_synsig_functions.csv')
	return final

final=compile_all_synsig_func()
print (final)
synsig_func_total=final.iloc[:, 3:18].sum()
final.loc['Total']=final.sum()
final.to_csv('all_synsig_functions.csv')

human_go=find_GO_scores.find_GO_ont()
go_genes=human_go.genes

big_pool=load_data_functions.load_big_pool()
syngo=load_data_functions.find_syngo(big_pool, go_genes)
syndb=load_data_functions.find_SynDB(big_pool)
synsysnet=load_data_functions.find_synsysnet(big_pool)

known=list(set(syngo+syndb+synsysnet))

synsig_genes=final['genes'].tolist()

overlap=list(set(synsig_genes)&set(known))
final=final.set_index('genes')
known_df=final.loc[overlap]
print (known_df)
known_df_total=known_df.iloc[:, 2:17].sum()
print (known_df_total)
known_df.loc['Total']=known_df.sum()
known_df.to_csv('known_synapse_functions.csv')

totals=pd.concat([synsig_func_total, known_df_total], axis=1)
print (totals)
totals.columns=['SynSig', 'Known']
print (totals)
totals['New']=totals['SynSig']-totals['Known']
print (totals)


x=list(totals.index)
known=totals['Known'].tolist()
new=totals['New'].tolist()
plt.bar(x, known, color='gray')
plt.bar(x, new, bottom=known, color='green')
plt.xticks(rotation=90)
plt.savefig('synsig_old_new_func'+'.svg', format="svg")
plt.close()






