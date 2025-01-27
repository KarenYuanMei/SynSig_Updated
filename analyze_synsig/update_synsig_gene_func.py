#Goal: to add description to predicted synapse gene (synsig) gene names
#gene_name_description.txt was downloaded on April 9th from Biomart

#to annotate each SynSig protein to a molecular function category
#need to run this on the cluster

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

def make_goID_dict():
	id_name=pd.read_table('../source_data_files/goID/goID_2_name.tab')
	print (id_name)

	id_name.columns=['ID', 'name']
	print (id_name)

	ids=id_name['ID'].tolist()
	names=id_name['name'].tolist()

	id_name_zip=list(zip(ids, names))
	d=defaultdict(list)

	for term, name in id_name_zip:
		d[term].append(name)
	return d

def find_function_cat(filename, variable, names, col_idx, cat_name, csv_filename):
	synsig=pd.read_csv(filename, index_col=[0])
	gene_desc=synsig[variable].tolist()

	functions=[]
	
	for item in gene_desc:
		names_in_desc = any(name in item for name in names)
		#print (names_in_desc)
		if names_in_desc==False:
			function=0
		else:
			function=1

		functions.append(function)

	num=functions.count(1)
	print (cat_name, num)

	synsig.insert(loc=col_idx, column=cat_name, value=functions)
	synsig.to_csv('%s'%csv_filename)
	return synsig


def make_gene_desc_dict():

	desc_df=pd.read_csv('../source_data_files/gene_names/gene_name_description.txt')

	#make gene_name: gene_description dictionary
	names=desc_df['Gene name'].tolist()
	desc=desc_df['Gene description'].tolist()
	names_desc=list(zip(names, desc))
	names_desc_dict=dict(names_desc)
	return names_desc_dict

def add_gene_desc_web(names_desc_dict, genelist):

	names=list(names_desc_dict.keys())

	synsig_desc_list=[]
	for item in genelist:
		if item in names:
			synsig_desc=names_desc_dict[item]
		else:
			synsig_desc='not found'
		synsig_desc_list.append(synsig_desc)

	idx=1

	#synsig_df.insert(loc=idx, column='description', value=synsig_desc_list)

	#synsig_df.to_csv('update_web_table_desc.csv')
	#synsig_df=pd.DataFrame({'genes': synsig_genes, 'description': synsig_desc_list})
	synsig_df=pd.DataFrame({'genes': genelist, 'description': synsig_desc_list})
	
	return synsig_df

def find_synsig_only(synsig_df):
	synsig=synsig_df[synsig_df['SynSig']=='yes']
	synsig.to_csv('synsig_only.csv')
	return synsig

def annotate_function(filename1, variable, filename2):
	synsig=find_function_cat(filename1, variable, ['receptor', 'channel'], 3, 'Receptor/Channel', filename2)
	synsig=find_function_cat(filename2, variable, ['kinase', 'phosphatase'], 4, 'Kinase/Phosphatase', filename2)
	synsig=find_function_cat(filename2, variable, ['ubiquitin', 'E3'], 5, 'Ubiquitin/E3', filename2)
	synsig=find_function_cat(filename2, variable, ['vesic', 'dynamin', 'clathrin', 'endoc', 'VAMP', 'SNAP', 'SNARE', 'translocase', 'export', 'import', 'transport', 'myosin', 'kinesin', 'dynein', 'dynactin'], 6, 'Vesicle/Transporters', filename2)
	synsig=find_function_cat(filename2, variable, ['GTPase', 'ATPase', 'exchange factor', 'GTP', 'ATP'], 7, 'GTP/ATP Regulators', filename2)
	synsig=find_function_cat(filename2, variable, ['DNA ', 'chromatin', 'transcription', 'nucleic acid', 'nucleotide'], 8, 'Nucleic Acid Binding', filename2)
	synsig=find_function_cat(filename2, variable, ['ribosom', 'translation', 'RNA', 'helicase'], 9, 'Translation', filename2)
	synsig=find_function_cat(filename2, variable, ['membrane', 'transmembrane', 'symporter', 'adhesion', 'cadherin', 'junction', 'junct', 'catenin'], 10, 'Membrane/Cell Adhesion', filename2)
	synsig=find_function_cat(filename2, variable, ['heat shock', 'regulator', 'chaperone'], 11, 'Other Regulators', filename2)
	synsig=find_function_cat(filename2, variable, ['scaffold', 'SHAN', 'assembl', 'adaptor'], 12, 'Scaffolds/Adaptors', filename2)
	synsig=find_function_cat(filename2, variable, ['microtubule', 'actin', 'filament', 'tubulin', 'filamin', 'cytoskelet'], 13, 'Cytoskeletal', filename2)
	synsig=find_function_cat(filename2, variable, ['calcium ion', 'calmodulin binding'], 14, 'Calcium Ion Binding', filename2)

	synsig['Function Total']= synsig.iloc[:, 2:15].sum(axis=1)

	func_total=synsig['Function Total'].tolist()
	synsig.pop('Function Total')
	synsig.insert(loc=15, column='Function Total', value=func_total)
	#print (synsig)
	synsig.to_csv(filename2)
	return synsig

#Annotate function by gene description: first wrapper function:
def annotate_function_by_desc():
	names_desc_dict=make_gene_desc_dict()
	synsig_df=add_gene_desc_web(names_desc_dict)
	synsig=find_synsig_only(synsig_df)

	filename1='synsig_only.csv'
	filename2='synsig_function.csv'

	synsig=annotate_function(filename1,'description', filename2)
	return synsig


def find_unannotated_genes(df):
	no_func=df[df['Function Total']==0]
	print (no_func)
	#no_func.to_csv('no_func.csv')
	no_func_genes=no_func['genes'].tolist()
	no_func_desc=no_func['description'].tolist()
	no_func_terms=no_func['MF Terms'].tolist()

	df=pd.DataFrame({'genes': no_func_genes, 'description': no_func_desc, 'MF Terms': no_func_terms})
	df.to_csv('no_func.csv')
	print (len(no_func_genes))
	return no_func_genes

def find_mf_ont():
	go_human=find_GO_scores.find_GO_ont()
	child_terms=go_human.parent_2_child['GO:00SUPER']
	mf=child_terms[0]
	ont2 = go_human.focus(branches=mf)
	return ont2

def add_mf_function(genelist, synsig_desc):
	ont2=find_mf_ont()
	terms=ont2.terms

	overlap=list(set(genelist)&set(ont2.genes))
	print (len(overlap))

	d=make_goID_dict()

	all_term_names=[]
	for gene in genelist:
		if gene in overlap:
			gene_term_names=[]
			gene_terms=ont2.gene_2_term[gene]
			for item in gene_terms:
				term_desc=terms[item]
				term_name=d[term_desc]
				gene_term_names.append(term_name)
			gene_term_names = [item for sublist in gene_term_names for item in sublist]

		else:
			gene_term_names='None'
		all_term_names.append(gene_term_names)

	synsig_desc.insert(loc=2, column='MF Terms', value=all_term_names)
	return synsig_desc

def add_desc(synsig_genes):
	names_desc_dict=make_gene_desc_dict()
	synsig_desc=add_gene_desc_web(names_desc_dict, synsig_genes)
	print (synsig_desc)
	return synsig_desc

#add description to all synsig genes:
def load_synsig_genes():
	synsig_df=pd.read_csv('../run_ML/update_web_table.csv', index_col=[0])
	#print (synsig_df)
	synsig=synsig_df[synsig_df['SynSig']=='yes']
	synsig_genes=synsig['genes'].tolist()
	return synsig_genes

def annotate_no_func(filename1, variable, filename2):
	synsig=find_function_cat(filename1, variable, ['enzyme', 'ase activity'], 3, 'Other Enzymes', filename2)
	synsig['Function Total']= synsig['Other Enzymes'].tolist()
	synsig.to_csv('%s'%filename2)
	return synsig

def annotate_remaining_func(filename1, variable, filename2):
	synsig=find_function_cat(filename1, variable, ['protein binding'], 3, 'Other protein binders', filename2)
	synsig=find_function_cat(filename2, variable, ['None'], 4, 'unknown functions', filename2)
	synsig['Function Total']= synsig.iloc[:, 3:5].sum(axis=1)
	synsig.to_csv('%s'%filename2)
	return synsig

def plot_bargraph(labels, mean_values, xlabel, ylabel, name):
	x_pos=np.arange(len(labels))
	#plt.bar(labels, mean_values, yerr=sem, color=['#7f6d5f', '#2d7f5e', '#557f2d','silver', 'dimgray', 'rosybrown'], align='center', ecolor='black', capsize=10)
	plt.bar(labels, mean_values, align='center', color='#2d7f5e', ecolor='black', capsize=10)

	#plt.ylim(1, 10**5)
	#plt.ylim(0.5, 1)
	#plt.yscale('log')
	# Create legend & Show graphic
	#plt.legend()
	#y_ticks = np.arange(0, 25, 5)
	#plt.yticks(y_ticks)
	plt.xlabel(xlabel, fontweight='bold')
	plt.ylabel(ylabel, fontweight='bold')
	plt.xticks(rotation=90)
	plt.savefig(name+'.svg', format="svg")
	plt.close()

def load_full_synsig_genes():

	df=pd.read_csv('~/Documents/SynSig_August2020/revisions/SynSig_Updated/supp_tables/updated_synsig_table_full_syngo_included_101222.csv')
	print (df)
	genes=df['genes'].tolist()

	return genes

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


if __name__ == '__main__':

	#synsig_genes=load_synsig_genes()
	print ('STEP 1')

	# synsig_genes=load_full_synsig_genes()
	# print (len(synsig_genes))
	# print (len(list(set(synsig_genes))))

	df=pd.read_csv('../supp_tables/updated_synsig_table_full_syngo_included_101222.csv')
	print (df)
	df=df.reset_index()
	print (df)
	df=df.rename(columns={ df.columns[1]: "genes" })
	
	synsig_genes=df['genes'].tolist()


	#add description all synsig genes:
	print ('STEP 2')
	synsig_desc=add_desc(synsig_genes)
	synsig_desc.to_csv('synsig_genes_desc.csv')

	#add mf to all synsig genes:
	print ('STEP 3')
	synsig_desc_mf=add_mf_function(synsig_genes, synsig_desc)
	synsig_desc_mf.to_csv('synsig_desc_mf.csv')
	print (synsig_desc_mf)

	#first annotate genes by gene description:
	print ('STEP 4')
	desc_func=annotate_function('synsig_desc_mf.csv', 'description', 'synsig_desc_function.csv')
	print (desc_func)

	#find unannotated genes:
	print ('STEP 5')
	no_func=find_unannotated_genes(desc_func)
	#print (no_func)

	#second annotate genes by molecular function:
	print ('STEP 6')
	mf_func=annotate_function('no_func.csv', 'MF Terms', 'synsig_mf_function.csv')
	print (mf_func)

	#find unannotated genes:
	print ('STEP 7')
	no_func=find_unannotated_genes(mf_func)

	met_func=annotate_no_func('no_func.csv', 'MF Terms', 'synsig_met_function.csv')
	print (met_func)

	#find unannotated genes:
	print ('STEP 8')
	no_func=find_unannotated_genes(met_func)

	unknown_func=annotate_remaining_func('no_func.csv', 'MF Terms', 'synsig_unknown_function.csv')
	print (unknown_func)

	desc_func=pd.read_csv('synsig_desc_function.csv', index_col=[0])
	mf_func=pd.read_csv('synsig_mf_function.csv', index_col=[0])

	cols=np.arange(3, 15, 1).tolist()
	print (cols)


	functions={}
	for item in cols:
		headers=list(desc_func.columns)
		function_name=headers[item]
		print (headers[item])
		sum1=desc_func[desc_func.columns[item]].sum()
		print (sum1)
		
		sum2=mf_func[mf_func.columns[item]].sum()
		print (sum2)
		
		total=sum1+sum2
		print (total)

		functions[function_name]=(total)

	print (functions)

	met_func=pd.read_csv('synsig_met_function.csv', index_col=[0])
	met_sum=met_func[met_func.columns[3]].sum()
	print (met_sum)

	uk_func=pd.read_csv('synsig_unknown_function.csv', index_col=[0])
	print (uk_func)
	pb_sum=uk_func[uk_func.columns[3]].sum()
	print (pb_sum)

	uk_sum=uk_func[uk_func.columns[4]].sum()
	print (uk_sum)

	functions['Other Enzymes']=met_sum
	functions['Other protein binders']=pb_sum
	functions['Unknown functions']=uk_sum

	print (functions)

	labels=list(functions.keys())
	mean_values=list(functions.values())
	xlabel='Protein Categories'
	ylabel='Number of Proteins'
	name='synsig_gene_cat'

	plot_bargraph(labels, mean_values, xlabel, ylabel, name)

	final=compile_all_synsig_func()
	print (final)
	synsig_func_total=final.iloc[:, 3:18].sum()
	final.loc['Total']=final.sum()
	final.to_csv('../supp_tables/all_synsig_functions.csv')






