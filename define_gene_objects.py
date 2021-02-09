#Goal: define gene objects and gene-pair objects and run random forest on five-fold cross-validation

import numpy as np
import pandas as pd
import csv
from scipy.stats.stats import pearsonr
from statistics import mean 

from collections import defaultdict
from itertools import combinations, combinations_with_replacement
from itertools import product
from scipy import spatial

#import seaborn as sns; sns.set()
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score
from scipy.stats.stats import pearsonr, spearmanr

from sklearn.kernel_approximation import PolynomialCountSketch

from sklearn.linear_model import SGDRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#import pylab
#from sklearn.datasets import make_regression
# import sys
# sys.path.append('../syngo_training')

from load_data_functions import get_gene_names

import sys
sys.path.append('../../../Network_propagation/Propagation_Code/')

#from make_network import make_mentha_df, make_network_G
#from load_genesets import find_seeds, find_nodesets, find_negatives
#from propagate_network import calculate_p, fast_random_walk, closed_form_network_propagation, construct_prop_kernel, normalize_network, get_propagated_scores


#kernel_df=pd.read_csv('../../../Network_propagation/propagate_synapse/mentha_kernel.csv', index_col=[0])
#create a dictionary gene_kernel_index: gene_name -> index in kernel df
#print (kernel_df)

#gtex_kernel_df=pd.read_csv('../new_features/gtex_rna_kernel/gtex_rna_kernel.csv', index_col=[0])
#print (gtex_kernel_df)

#full 60 features:
def define_features():
	feature_list=['cerebral_cortex_hpa_isoform_exp', 'colon_hpa_isoform_exp', 'ovary_hpa_isoform_exp', 'breast_hpa_isoform_exp', 'lung_hpa_isoform_exp', 'salivary gland_hpa_isoform_exp', 'seminal vesicle_hpa_isoform_exp', 
		'lymph node_hpa_isoform_exp', 'placenta_hpa_isoform_exp', 'kidney_hpa_isoform_exp', 'cervix, uterine_hpa_isoform_exp', 'adrenal gland_hpa_isoform_exp', 'thyroid gland_hpa_isoform_exp', 
		'stomach 1_hpa_isoform_exp', 'gallbladder_hpa_isoform_exp', 'duodenum_hpa_isoform_exp', 'fallopian tube_hpa_isoform_exp','endometrium 1_hpa_isoform_exp', 'skin 1_hpa_isoform_exp', 
		'spleen_hpa_isoform_exp', 'gtex_rna_tissue_expression', 'appendix_hpa_isoform_exp', 'heart muscle_hpa_isoform_exp', 'small intestine_hpa_isoform_exp', 'epididymis_hpa_isoform_exp', 'testis_hpa_isoform_exp', 
		'liver_hpa_isoform_exp', 'esophagus_hpa_isoform_exp', 'urinary bladder_hpa_isoform_exp', 'skeletal muscle_hpa_isoform_exp', 'tonsil_hpa_isoform_exp', 'prostate_hpa_isoform_exp', 
		'parathyroid gland_hpa_isoform_exp','adipose tissue_hpa_isoform_exp', 'smooth muscle_hpa_isoform_exp', 'rectum_hpa_isoform_exp', 'bone marrow_hpa_isoform_exp', 'mentha_source_feature', 
		'chr_no_source_feature', 'qPhos_site_number', 'Phosphosite_hu_no', 'pFAM_domain_number', 'pFAM_domain', 'protein_mass', 'Ensembl_aa_length', 'Ensembl_isoform_no', 
		'trans_count', 'gc_content', 'trans_len', 'gene_length', 'exon_no', 'cds_length']

	brain_features=['HIP_RNA', 'DFC_RNA', 'V1C_RNA', 'AMY_RNA', 'MD_RNA', 'STR_RNA', 'CBC_RNA']

	kernel_feature=['mentha_kernel']
	feature_list=feature_list+brain_features+kernel_feature
	#print (len(list(set(feature_list))))
	return feature_list


# def define_features():
# 	feature_list=['gtex_rna_tissue_expression',  'testis_hpa_isoform_exp','mentha_source_feature', 'qPhos_site_number', 'Phosphosite_hu_no', 'pFAM_domain_number', 'pFAM_domain', 'protein_mass', 'Ensembl_aa_length', 'Ensembl_isoform_no', 'trans_count', 'gc_content', 'trans_len', 'gene_length', 'exon_no', 'cds_length']
# 	brain_features=['HIP_RNA', 'DFC_RNA', 'V1C_RNA', 'AMY_RNA', 'MD_RNA', 'STR_RNA', 'CBC_RNA']

# 	kernel_feature=['mentha_kernel']
# 	feature_list=feature_list+brain_features+kernel_feature
# 	#feature_list.pop(idx)

# 	return feature_list


# #few_plus_one_features
# def define_features():
# 	feature_list=['cerebral_cortex_hpa_isoform_exp', 'gtex_rna_tissue_expression','Phosphosite_hu_no', 'pFAM_domain_number', 'pFAM_domain', 'protein_mass', 'Ensembl_aa_length', 'Ensembl_isoform_no', 'trans_count', 'gc_content', 'trans_len', 'gene_length', 'exon_no', 'cds_length']

# 	brain_features=['HIP_RNA', 'DFC_RNA', 'V1C_RNA', 'AMY_RNA', 'MD_RNA', 'STR_RNA', 'CBC_RNA']
# 	kernel_feature=['mentha_kernel']
# 	#gtex_kernel_feature=['gtex_rna_kernel']
# 	feature_list=feature_list+brain_features+kernel_feature
# 	#feature_list.pop(idx)

# 	return feature_list

#only DFC
# def define_features():
# 	feature_list=['cerebral_cortex_hpa_isoform_exp', 'gtex_rna_tissue_expression','Phosphosite_hu_no', 'pFAM_domain_number', 'pFAM_domain', 'protein_mass', 'Ensembl_aa_length', 'Ensembl_isoform_no', 'trans_count', 'gc_content', 'trans_len', 'gene_length', 'exon_no', 'cds_length']

# 	brain_features=['HIP_RNA', 'DFC_RNA']
# 	kernel_feature=['mentha_kernel']
# 	#gtex_kernel_feature=['gtex_rna_kernel']
# 	feature_list=feature_list+brain_features+kernel_feature
# 	#feature_list.pop(idx)

# 	return feature_list

class Gene:
	def __init__(self,name,is_test_gene):
		assert isinstance(name,str), "You screwed up! The name you gave is not a string!"
		assert isinstance(is_test_gene,bool), "You screwed up! is_test_gene needs to be True or False!"

		self.name = name
		self.is_test_gene = is_test_gene

	def create_feature(self,feature_name,feature_value):
		self.__dict__[feature_name] = feature_value

	def create_GO_scores(self, GO_scores):
		#GO_scores is a dictionary with th gene2_names as key and the go_score as the value
		self.go_scores = GO_scores

class PairOfGenes:
	def __init__(self,gene1,gene2,include_GO=True):
		assert isinstance(gene1,Gene), "You screwed up! gene1 needs to be a Gene!"
		assert isinstance(gene2,Gene), "You screwed up! gene2 needs to be a Gene!"

		self.gene1_name = gene1.name
		self.gene2_name = gene2.name

		feature_list=define_features()
#find the genes in all of the features:

		for feature_name in feature_list:
			gene1_feature = gene1.__dict__[feature_name]
			gene2_feature = gene2.__dict__[feature_name]
			if self.check_for_missing_feature(gene1_feature, gene2_feature) == "Missing":
				self.__dict__[feature_name] = 0
			else:
				self.combine_features(gene1_feature,gene2_feature,feature_name)


		if include_GO:
			self.create_pair_GO_score(gene1, gene2)

	def check_for_missing_feature(self, gene1_feature,gene2_feature):
		#check if numpy array:
		#then check if all values are 0
		#check if not numpy array:
		#then check if values are 0
		if type(gene1_feature) is np.ndarray or type(gene2_feature) is np.ndarray:
			if np.all(gene1_feature==0) == True or np.all(gene2_feature==0)==True:
				return "Missing"
		else:
			if gene1_feature ==0 or gene2_feature == 0:
				return "Missing"
			
	def combine_features(self,gene1_feature,gene2_feature,feature_name):
	
		pearson_features = ['cerebral_cortex_hpa_isoform_exp', 'colon_hpa_isoform_exp', 'ovary_hpa_isoform_exp', 'breast_hpa_isoform_exp', 'lung_hpa_isoform_exp', 'salivary gland_hpa_isoform_exp', 'seminal vesicle_hpa_isoform_exp',
		'lymph node_hpa_isoform_exp', 'placenta_hpa_isoform_exp', 'kidney_hpa_isoform_exp', 'cervix, uterine_hpa_isoform_exp', 'adrenal gland_hpa_isoform_exp', 'thyroid gland_hpa_isoform_exp', 
		'stomach 1_hpa_isoform_exp', 'gallbladder_hpa_isoform_exp', 'duodenum_hpa_isoform_exp', 'fallopian tube_hpa_isoform_exp','endometrium 1_hpa_isoform_exp', 'skin 1_hpa_isoform_exp', 
		'spleen_hpa_isoform_exp', 'gtex_rna_tissue_expression', 'appendix_hpa_isoform_exp', 'heart muscle_hpa_isoform_exp', 'small intestine_hpa_isoform_exp', 'epididymis_hpa_isoform_exp', 'testis_hpa_isoform_exp',
		'liver_hpa_isoform_exp', 'esophagus_hpa_isoform_exp', 'urinary bladder_hpa_isoform_exp', 'skeletal muscle_hpa_isoform_exp', 'tonsil_hpa_isoform_exp', 'prostate_hpa_isoform_exp', 
		'parathyroid gland_hpa_isoform_exp','adipose tissue_hpa_isoform_exp', 'smooth muscle_hpa_isoform_exp', 'rectum_hpa_isoform_exp', 'bone marrow_hpa_isoform_exp', 
		'HIP_RNA', 'DFC_RNA', 'V1C_RNA', 'AMY_RNA', 'MD_RNA', 'STR_RNA', 'CBC_RNA']
		subtraction_features=['Phosphosite_hu_no', 'qPhos_site_number', 'Ensembl_isoform_no', 'Ensembl_aa_length', 'pFAM_domain_number', 'protein_mass', "trans_count", 'gc_content', 'trans_len', 'gene_length', 'exon_no', 'cds_length']
		jaccard_features=['pFAM_domain', 'mentha_source_feature']
	
		if feature_name in pearson_features:
			self.__dict__[feature_name] = pearsonr(gene1_feature,gene2_feature)[0]

		elif feature_name in subtraction_features:
			self.__dict__[feature_name]=abs(gene1_feature-gene2_feature)[0]

		elif feature_name in jaccard_features:
			no1=set(gene1_feature)
			no2=set(gene2_feature)
			self.__dict__[feature_name]=len(no1.intersection(no2)) / len(no1.union(no2))

		elif feature_name=="chr_no_source_feature":
			if gene1_feature == gene2_feature:
				result=1
				self.chr_no_source_feature=result
			else:
				result=0
				self.chr_no_source_feature=result

		elif feature_name == 'mentha_kernel':
			row=self.gene1_name
			col=self.gene2_name
			value1=kernel_df.loc[row, col]
			value2=kernel_df.loc[col, row]
			avg=mean([value1, value2])
			#print ('mentha kernel', avg)
			self.__dict__[feature_name]=avg

		elif feature_name == 'gtex_rna_kernel':
			row=self.gene1_name
			col=self.gene2_name
			value1=gtex_kernel_df.loc[row, col]
			value2=gtex_kernel_df.loc[col, row]
			avg=mean([value1, value2])
			#print ('mentha kernel', avg)
			self.__dict__[feature_name]=avg

	def create_pair_GO_score(self, gene1, gene2):
		gene2_name=gene2.name
		GO_score=gene1.go_scores[gene2_name]
		self.GO_score = GO_score

def find_input_features(filename, input_genes):
	string_files=['pFAM_domain', 'mentha_source_feature','biogrid_source_feature', 'bioplex_source_feature', 'chr_no_source_feature']
	kernel_file=['mentha_kernel']
	gtex_rna_file=['gtex_rna_kernel']
	if filename in string_files:
		df = pd.read_csv('../../SynSig/features/normalized_%s.csv'%filename,converters={"Interactors": lambda x: x.strip("[]").split(", ")})
		symbol=df['Norm_Symbol']
		df.drop(labels=['Norm_Symbol', 'Genes'], axis=1,inplace = True)
		df.insert(0, 'Genes', symbol)
		df=df.set_index('Genes')
		df=df.loc[input_genes]

	elif filename in kernel_file:
		df=pd.read_csv('../../../Network_propagation/propagate_synapse/mentha_kernel.csv', index_col=[0])
		df=df.loc[input_genes]

	elif filename in gtex_rna_file:
		df=pd.read_csv('../new_features/gtex_rna_kernel/gtex_rna_kernel.csv', index_col=[0])
		df=df.loc[input_genes]

	else:
		df=pd.read_csv('../../SynSig/features/normalized_%s.csv'%filename)

		symbol=df['Norm_Symbol']
		df.drop(labels=['Norm_Symbol', 'Genes'], axis=1,inplace = True)
		df.insert(0, 'Genes', symbol)
		#print (df)
		df=df.set_index('Genes')
		df=df.loc[input_genes]
		
	return df

def load_feature(filename, input_genes):
	feature=find_input_features(filename, input_genes)
	string_files=['pFAM_domain', 'mentha_source_feature','biogrid_source_feature', 'bioplex_source_feature', 'chr_no_source_feature']
	kernel_file=['mentha_kernel']
	gtex_rna_file=['gtex_rna_kernel']
	feature=feature.fillna(0)
	idx=list(feature.index)
	#values=feature.values
	if filename in string_files:
		#print ('TRUE')
		values=feature[feature.columns[0]]
		values=values.tolist()
		feature_dict=dict(list(zip(idx, values)))

	elif filename in kernel_file:
		genes=list(feature.columns)
		index_no=list(range(len(genes)))

		gene_index=list(zip(genes,index_no))
		feature_dict=dict(gene_index)

	elif filename in gtex_rna_file:
		genes=list(feature.columns)
		index_no=list(range(len(genes)))

		gene_index=list(zip(genes,index_no))
		feature_dict=dict(gene_index)

	else:
		#in case of duplicate values, do a defaultdict
		values=feature.values
		idx_values=list(zip(idx, values))
		d=defaultdict(list)
		for idx, value in idx_values:
			d[idx].append(value)
		new_values=[]
		keys=[]
		for key in d:
			feature_values=d[key]
			keys.append(key)
			new_value=np.mean(feature_values, axis=0)
			new_values.append(new_value)
		feature_dict=dict(zip(keys, new_values))
		#print (feature_dict['KCNMA1'])
	return feature_dict

def create_feature_value_dict(input_genes):
	feature_list=define_features()
#find the genes in all of the features:

	all_feature_values=[]
	items=[]
	for item in feature_list:
		feature_values=load_feature(item, input_genes)
		items.append(item)
		all_feature_values.append(feature_values)

	feature_dict=dict(zip(items, all_feature_values))
	return feature_dict

def get_feature_value(gene_name, feature_name, feature_value_dict):
	#returns the feature value for a specific gene
	feature_values_for_all_genes = feature_value_dict[feature_name]
	feature_value = feature_values_for_all_genes[gene_name]
	return feature_value
	
#go_mat_filename='syngo_GO_training_score_matrix_for_big_pool_genes.csv'

def create_GO_score_dict(go_score_mat):
	#df=pd.read_csv(go_mat_filename, index_col=[0])
	df=go_score_mat
	
	idx=list(df.index)
	cols=list(df.columns)

	all_dict=[]
	for gene1 in idx:
		gene1_scores=[]
		for gene2 in cols:
			score=df.loc[gene1, gene2]
			gene1_scores.append(score)
		gene2_dict=dict(zip(cols, gene1_scores))
		all_dict.append(gene2_dict)
	
	master_dict=dict(zip(idx, all_dict))
	#print (master_dict['STX4'])
	return master_dict

def create_gene_list(gene_names,is_test_gene,feature_value_dict, go_score_mat):
	#returns a list of Gene objects, corresponding to the names in gene_names
	#feature_value_dict is a dictionary containing all feature values for all genes

	feature_list=define_features()
	gene_list = []
	for name in gene_names:
		new_gene = Gene(name, is_test_gene)
		for feature_name in feature_list:
			feature_value = get_feature_value(name,feature_name,feature_value_dict)
			new_gene.create_feature(feature_name, feature_value)
		gene_list.append(new_gene)

	GO_score_dict=create_GO_score_dict(go_score_mat)

	for gene1 in gene_list:
		gene1_name =gene1.name
		go_scores=GO_score_dict[gene1_name]
		gene1.create_GO_scores(go_scores)

	return gene_list

def create_data_gene_list(gene_names,is_test_gene,feature_value_dict):
	feature_list=define_features()
	gene_list = []
	for name in gene_names:
		new_gene = Gene(name, is_test_gene)
		for feature_name in feature_list:
			feature_value = get_feature_value(name,feature_name,feature_value_dict)
			new_gene.create_feature(feature_name, feature_value)
		gene_list.append(new_gene)
	return gene_list

def find_pos_genes_in_training(training_genes, positives):
	overlap=list(set(training_genes)&set(positives))
	input_genes=list(set(overlap))
	#print (len(input_genes))
	return input_genes

def find_gene_objects(gene_objects, gene_name_list):
	sel_objects=[]
	for item in gene_objects:
		gene_name=item.name
		#print (gene_name)
		if gene_name in gene_name_list:
			sel_objects.append(item)
		else:
			continue
	return sel_objects

def find_feature_array(feature_list, pair_object):
	pair_feature_array=[]
	for feature_name in feature_list:
			#print (feature_name)
		pair_feature_values=pair_object.__dict__[feature_name]
		pair_feature_array.append(pair_feature_values)
	return pair_feature_array

def create_input_pair_objects(gene_pairs):
	feature_list=define_features()
	#feature_list.pop(idx)

	gene_pair_objects=[]
	feature_array=[]
	score_array=[]
	for item in gene_pairs:
		gene1=item[0]
		gene2=item[1]
		pair_object=PairOfGenes(gene1, gene2)
		
		pair_feature_array=find_feature_array(feature_list, pair_object)
		feature_array.append(pair_feature_array)
		
		pair_GO_score=pair_object.GO_score
		score_array.append(pair_GO_score)

		gene_pair_objects.append(pair_object)

	feature_array=np.array(feature_array)
	score_array=np.array(score_array)
	return gene_pair_objects, feature_array, score_array

def find_gene1_gene2(train_test_gene_pair_objects):
	genes_1=[]
	genes_2=[]
	for item in train_test_gene_pair_objects:
		gene1=item.gene1_name
		genes_1.append(gene1)
		gene2=item.gene2_name
		genes_2.append(gene2)
	return genes_1, genes_2

def run_random_forest(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, number):
	
	#training_gene_pair_objects, training_feature_array, training_score=create_input_pair_objects(training_pairs)

	#print ('DONE')
	#train_test_gene_pair_objects, tt_feature_array, tt_score=create_input_pair_objects(train_test_pairs)
	#print ('DONE')
	#training_gene_pair_objects, train_test_gene_pair_objects=load_objects()

	X_train=training_feature_array
	print (X_train.shape)
	X_test=tt_feature_array

	y_train=training_score
	print (y_train.shape)
	y_test=tt_score

	print ('X_test', len(X_test), 'y_test', len(y_test))

	#forest = RandomForestRegressor(n_estimators=100, max_depth=50, oob_score=True, random_state=0)
	forest = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=0)
	#forest = RandomForestRegressor(200)
	forest.fit(X_train, y_train)
	#depth=[(est.get_depth(), est.tree_.max_depth, est.max_depth) for est in forest.estimators_]
	#print (depth)

	yfit=forest.predict(X_test)

	print (len(yfit))
	print (len(y_test))

	performance=forest.feature_importances_
	performance=performance.tolist()

#metrics:
	print ('feature importance', performance)
	print ('train score', forest.score(X_train, y_train))
	print ('oob_sore', forest.oob_score_)
	print ('test score:', forest.score(X_test, y_test))
	print ('explained variance score', explained_variance_score(y_test, yfit))
	print ('mean absolute error', mean_absolute_error(y_test, yfit))
	print ('r^2 score', r2_score(y_test, yfit))
	print ('ytest', y_test)
	yfit=np.array(yfit)
	print ('yfit', yfit)

	feature_list=define_features()

	perf=pd.DataFrame({'Features': feature_list, 'Importance': performance})
	perf.to_csv('full60_random_forest_Feature_Importance_%s.csv'%number)
#find the genes in all of the features:

	df=pd.DataFrame({'ypredict':yfit, 'ytest': y_test})
	index=df.index

	print (spearmanr(y_test, yfit))
	print (pearsonr(y_test, yfit))
	pearson_corr=pearsonr(y_test, yfit)[0]
	pearson_corr=np.round(pearson_corr,2)
	p_value=pearsonr(y_test, yfit)[1]

	genes_1, genes_2=find_gene1_gene2(train_test_gene_pair_objects)

	df['Gene1']=genes_1
	df['Gene2']=genes_2

	df = df[['Gene1', 'Gene2', 'ytest', 'ypredict']]
	print (df)

	#df.to_csv('/Users/karenmei/Documents/Synapse_Ontology/NetworkCla/Entry_Ontology/synapse_10/random_forest/ypredict_ytest_%s.csv'%number)
	df.to_csv('../add_feature/regressors/full60_random_forest_%s.csv'%number)

		# yfit=forest.predict(X_train)
		# yfit=np.array(yfit)
		# df=pd.DataFrame({'ypredict_train':yfit, 'yactual': y_train})

		# genes_1=[]
		# genes_2=[]
		# for item in training_gene_pair_objects:
		# 	gene1=item.gene1_name
		# 	genes_1.append(gene1)
		# 	gene2=item.gene2_name
		# 	genes_2.append(gene2)
		# df['Gene1']=genes_1
		# df['Gene2']=genes_2
		# df = df[['Gene1', 'Gene2', 'yactual', 'ypredict_train']]
		# print (df)

		# df.to_csv('%s_train_removed_features_%s_treeno_%s_classifier.csv'%(name, number, tree_no))
	return df



def run_adaboost(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, number):
	X_train=training_feature_array
	print (X_train.shape)
	X_test=tt_feature_array

	y_train=training_score
	print (y_train.shape)
	y_test=tt_score
	regr = AdaBoostRegressor(random_state=0, n_estimators=100)
	regr.fit(X_train, y_train)

	yfit=regr.predict(X_test)

	yfit=np.array(yfit)
	print ('yfit', yfit)

	df=pd.DataFrame({'ypredict':yfit, 'ytest': y_test})
	index=df.index

	print (spearmanr(y_test, yfit))
	print (pearsonr(y_test, yfit))

	# genes_1=[]
	# genes_2=[]
	# for item in train_test_gene_pair_objects:
	# 	gene1=item.gene1_name
	# 	genes_1.append(gene1)
	# 	gene2=item.gene2_name
	# 	genes_2.append(gene2)
	genes_1, genes_2=find_gene1_gene2(train_test_gene_pair_objects)

	df['Gene1']=genes_1
	df['Gene2']=genes_2

	df = df[['Gene1', 'Gene2', 'ytest', 'ypredict']]
	print (df)

	df.to_csv('full60_adaboost_%s.csv'%number)
	return df

def run_svm_regressor(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, number):
	X_train=training_feature_array
	print (X_train.shape)
	X_test=tt_feature_array

	y_train=training_score
	print (y_train.shape)
	y_test=tt_score

	regr = make_pipeline(StandardScaler(), SVR(kernel='sigmoid', C=1.0, epsilon=0.2))
	regr.fit(X_train, y_train)
	
	yfit=regr.predict(X_test)

	# ps = PolynomialCountSketch(degree=4, random_state=0)
	# X_features=ps.fit_transform(X_train)
	# regr = SGDRegressor(tol=1e-3)
	# regr.fit(X_features, y_train)

	# X_test=ps.fit_transform(X_test)

	# yfit=regr.predict(X_test)

	# yfit=np.array(yfit)
	# print ('yfit', yfit)

	df=pd.DataFrame({'ypredict':yfit, 'ytest': y_test})
	index=df.index

	print (spearmanr(y_test, yfit))
	print (pearsonr(y_test, yfit))

	# genes_1=[]
	# genes_2=[]
	# for item in train_test_gene_pair_objects:
	# 	gene1=item.gene1_name
	# 	genes_1.append(gene1)
	# 	gene2=item.gene2_name
	# 	genes_2.append(gene2)
	genes_1, genes_2=find_gene1_gene2(train_test_gene_pair_objects)

	df['Gene1']=genes_1
	df['Gene2']=genes_2

	df = df[['Gene1', 'Gene2', 'ytest', 'ypredict']]
	print (df)

	df.to_csv('../add_feature/regressors/full60_svregressor_sigmoid_%s.csv'%number)


def find_data_genes(training_genes):
	#new_index=pd.read_csv('/Users/karenmei/Documents/Synapse_Ontology/NetworkClass/Entry_Ontology/synapse_10/no_brain_genes_index.csv')

	new_index=pd.read_csv('../../SynSig/synsig_random_forest/big_pool_genes_index.csv')
	all_genes=new_index['genes'].tolist()
	data_genes=list(set(all_genes)-set(training_genes))
	return data_genes



def find_data_array(gene_pairs, feature_list):
	#gene_pair_objects=[]
	gene1_all=[]
	gene2_all=[]
	feature_array=[]
	for item in gene_pairs:
		gene1=item[0]
		gene2=item[1]
		pair_object=PairOfGenes(gene1, gene2, include_GO=False)
		
		pair_feature_array=find_feature_array(feature_list, pair_object)
		feature_array.append(pair_feature_array)

		gene1=pair_object.gene1_name
		gene1_all.append(gene1)

		gene2=pair_object.gene2_name
		gene2_all.append(gene2)
		
		#gene_pair_objects.append(pair_object)

	feature_array=np.array(feature_array)
	return feature_array, gene1_all, gene2_all


def run_new_rf(X_train, y_train, data_test, data_gene1, data_gene2):

	forest = RandomForestRegressor(n_estimators=100, max_depth=50, oob_score=True, random_state=0)
	#forest = RandomForestRegressor(200)
	forest.fit(X_train, y_train)

	#------actual new genes-------------------------------------------------------------------------------------------------------------
	print (data_test.shape)
	print (data_test)

	print('nan', np.isnan(data_test).any())
	nan_idx=np.where(np.isnan(data_test))
	data_test[nan_idx]=0
	print(np.where(np.isnan(data_test)))
	print('infinity', np.isfinite(data_test).all())

	data_fit=forest.predict(data_test)

	print (len(data_fit))

	df=pd.DataFrame({'ypredict':data_fit})

	df['Gene1']=data_gene1
	df['Gene2']=data_gene2

	df = df[['Gene1', 'Gene2', 'ypredict']]
	print (df)
	df.to_csv('new_all_gene_predictions.csv')

def find_avg_scores(data_genes):
	
	print ('data_genes', len(data_genes))

	pred_filename='new_all_gene_predictions.csv'
	pred=pd.read_csv(pred_filename, index_col=[0])
	print (pred)
	gene1=pred['Gene1'].tolist()
	gene2=pred['Gene2'].tolist()

	avg_scores=[]
	novel_genes=[]
	for gene in data_genes:
		df1=pred.loc[pred['Gene1'] == gene]
		
		df2=pred.loc[pred['Gene2'] == gene]
		df=df1.append(df2)
		scores=df['ypredict'].tolist()
		print ('length of scores', len(scores))
		scores_np=np.array(scores)
		avg_score=np.mean(scores_np)
		
		avg_scores.append(avg_score)
		novel_genes.append(gene)

	print ('novel_genes', len(novel_genes), 'all_average_scores', len(avg_scores))

	df=pd.DataFrame({'genes': novel_genes, 'avg_scores': avg_scores})
	df.to_csv('new_brain_RNA_big_pool_novel_synapse_genes_avg_scores.csv')
	return df

