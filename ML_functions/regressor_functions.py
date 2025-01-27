import numpy as np
import pandas as pd
import csv
from scipy.stats.stats import pearsonr
from statistics import mean 

from itertools import combinations, combinations_with_replacement
from itertools import product

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

from sklearn.inspection import permutation_importance

#from define_gene_objects import define_features

def find_gene1_gene2(train_test_gene_pair_objects):
	genes_1=[]
	genes_2=[]
	for item in train_test_gene_pair_objects:
		gene1=item.gene1_name
		genes_1.append(gene1)
		gene2=item.gene2_name
		genes_2.append(gene2)
	return genes_1, genes_2

def make_pred_df(yfit, y_test, train_test_gene_pair_objects):
	df=pd.DataFrame({'ypredict':yfit, 'ytest': y_test})
	genes_1, genes_2=find_gene1_gene2(train_test_gene_pair_objects)
	df['Gene1']=genes_1
	df['Gene2']=genes_2
	df = df[['Gene1', 'Gene2', 'ytest', 'ypredict']]
	print (df)
	return df


def redefine_input(training_feature_array, tt_feature_array, training_score, tt_score):
	X_train=training_feature_array
	print (X_train.shape)
	X_test=tt_feature_array

	y_train=training_score
	print (y_train.shape)
	y_test=tt_score
	return X_train, X_test, y_train, y_test

def print_pred_metrics(predictor, X_train, X_test, y_train, y_test, yfit):
	print ('train score', predictor.score(X_train, y_train))
	print ('oob_sore', predictor.oob_score_)
	print ('test score:', predictor.score(X_test, y_test))
	print ('explained variance score', explained_variance_score(y_test, yfit))
	print ('mean absolute error', mean_absolute_error(y_test, yfit))
	print ('r^2 score', r2_score(y_test, yfit))

	print (spearmanr(y_test, yfit))
	print (pearsonr(y_test, yfit))
	spearmanr_corr=spearmanr(y_test, yfit)[0]
	spearmanr_corr=np.round(spearmanr_corr,2)
	p_value=spearmanr(y_test, yfit)[1]
	return spearmanr_corr, p_value

def find_feature_importance(predictor, feature_list, name):
	performance=predictor.feature_importances_
	performance=performance.tolist()
	#feature_list=define_features()
	perf=pd.DataFrame({'Features': feature_list, 'Importance': performance})
	perf.to_csv('%s_random_forest_Feature_Importance.csv'%name)
	return perf

def find_permutation_importance(predictor, X_test, y_test):
	#use this function to find feature importance
	#predictor is the trained random forest
	#X_test is the features for the test set genes
	#y_test is the ground-truth labels for the test set genes

	importance = permutation_importance(predictor, X_test, y_test, n_repeats=50, random_state=0)
	print (importance)

	with open('synsig_permutation_imp.csv', 'w') as csv_file:  
	    writer = csv.writer(csv_file)
	    for key, value in importance.items():
	       writer.writerow([key, value])

	#importance is a dictionary: feature_name -> importance
	return importance

def run_random_forest(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, feature_list, number):

	#X_train, X_test, y_train, y_test=redefine_input(training_feature_array, tt_feature_array, training_score, tt_score)
	print ('X_test', len(X_test), 'y_test', len(y_test))

	#forest = RandomForestRegressor(n_estimators=100, max_depth=50, oob_score=True, random_state=0)
	forest = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=0)
	#forest = RandomForestRegressor(200)
	forest.fit(X_train, y_train)
	#depth=[(est.get_depth(), est.tree_.max_depth, est.max_depth) for est in forest.estimators_]
	#print (depth)
	yfit=forest.predict(X_test)
	yfit=np.array(yfit)
	print ('yfit', yfit)

#metrics:
	spearmanr_corr, p_value=print_pred_metrics(forest, X_train, X_test, y_train, y_test, yfit)
	#print ('ytest', y_test)
	feature_imp=find_feature_importance(forest, number, feature_list)

#find the genes in all of the features:
	df=make_pred_df(yfit, y_test, train_test_gene_pair_objects)
	#df.to_csv('/Users/karenmei/Documents/Synapse_Ontology/NetworkCla/Entry_Ontology/synapse_10/random_forest/ypredict_ytest_%s.csv'%number)
	df.to_csv('../run_ML/ML_output/regressors/full60_random_forest_%s.csv'%number)

	return df

def run_adaboost(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, number):
	X_train, X_test, y_train, y_test=redefine_input(training_feature_array, tt_feature_array, training_score, tt_score)

	regr = AdaBoostRegressor(random_state=0, n_estimators=100)
	regr.fit(X_train, y_train)

	yfit=regr.predict(X_test)

	yfit=np.array(yfit)
	print ('yfit', yfit)

	df=pd.DataFrame({'ypredict':yfit, 'ytest': y_test})
	index=df.index

	print (spearmanr(y_test, yfit))
	print (pearsonr(y_test, yfit))

	df=make_pred_df(yfit, y_test, train_test_gene_pair_objects)
	df.to_csv('../run_ML/ML_output/regressors/full60_adaboost_%s.csv'%number)
	return df

def run_svm_regressor(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, kernel, number):

	X_train, X_test, y_train, y_test=redefine_input(training_feature_array, tt_feature_array, training_score, tt_score)

	regr = make_pipeline(StandardScaler(), SVR(kernel=kernel, C=1.0, epsilon=0.1))
	regr.fit(X_train, y_train)
	
	yfit=regr.predict(X_test)

	print (spearmanr(y_test, yfit))
	print (pearsonr(y_test, yfit))

	df=make_pred_df(yfit, y_test, train_test_gene_pair_objects)

	df.to_csv('../run_ML/ML_output/regressors/full60_svregressor_%s_%s.csv'%(kernel, number))


def run_svm_poly(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, poly_number, number):
	X_train, X_test, y_train, y_test=redefine_input(training_feature_array, tt_feature_array, training_score, tt_score)

	ps = PolynomialCountSketch(degree=poly_number, random_state=0)
	X_features=ps.fit_transform(X_train)
	regr = SGDRegressor(tol=1e-3)
	regr.fit(X_features, y_train)

	X_test=ps.fit_transform(X_test)

	yfit=regr.predict(X_test)

	yfit=np.array(yfit)
	print ('yfit', yfit)

	print (spearmanr(y_test, yfit))
	print (pearsonr(y_test, yfit))

	df=make_pred_df(yfit, y_test, train_test_gene_pair_objects)

	df.to_csv('../run_ML/ML_output/regressors/full60_svregressor_%s_%s.csv'%(poly_number, number))


def sweep_param_rf(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, tree_no, depth, split):

	#X_train, X_test, y_train, y_test=redefine_input(training_feature_array, tt_feature_array, training_score, tt_score)
	print ('X_test', len(X_test), 'y_test', len(y_test))

	#forest = RandomForestRegressor(n_estimators=100, max_depth=50, oob_score=True, random_state=0)
	forest = RandomForestRegressor(n_estimators=tree_no, max_depth=depth, min_samples_split=split, oob_score=True, random_state=0)
	#forest = RandomForestRegressor(200)
	forest.fit(X_train, y_train)
	#depth=[(est.get_depth(), est.tree_.max_depth, est.max_depth) for est in forest.estimators_]
	#print (depth)
	yfit=forest.predict(X_test)
	yfit=np.array(yfit)
	print ('yfit', yfit)

#metrics:
	#spearmanr_corr, p_value=print_pred_metrics(forest, X_train, X_test, y_train, y_test, yfit)
	#print ('ytest', y_test)
	#feature_imp=find_feature_importance(forest, number, feature_list)

#find the genes in all of the features:
	df=make_pred_df(yfit, y_test, train_test_gene_pair_objects)
	#df.to_csv('/Users/karenmei/Documents/Synapse_Ontology/NetworkCla/Entry_Ontology/synapse_10/random_forest/ypredict_ytest_%s.csv'%number)
	#df.to_csv('../run_ML/ML_output/regressors/full60_random_forest_%s.csv'%number)

	return df


#optimal parameters: n_estimators=100, max_depth=50
def run_new_rf(X_train, y_train, new_test, new_gene1, new_gene2, tree_no, depth, split):

	forest = RandomForestRegressor(n_estimators=tree_no, max_depth=depth, min_samples_split=split, oob_score=True, random_state=0)
	#forest = RandomForestRegressor(200)
	forest.fit(X_train, y_train)

	#------actual new genes-------------------------------------------------------------------------------------------------------------
	print (new_test.shape)
	print (new_test)

	print('nan', np.isnan(new_test).any())
	nan_idx=np.where(np.isnan(new_test))
	new_test[nan_idx]=0
	print(np.where(np.isnan(new_test)))
	print('infinity', np.isfinite(new_test).all())

	print (new_test.shape)

	new_fit=forest.predict(new_test)



	print (len(new_fit))

	df=pd.DataFrame({'ypredict':new_fit})

	df['Gene1']=new_gene1
	df['Gene2']=new_gene2

	df = df[['Gene1', 'Gene2', 'ypredict']]
	print (df)
	#df.to_csv('updated_new_all_gene_predictions.csv')
	return forest, df

# def find_avg_scores(pred_filename, new_genes, name):
	
# 	print ('new_genes', len(new_genes))

# 	#pred_filename='new_all_gene_predictions.csv'
# 	pred=pd.read_csv(pred_filename, index_col=[0])
# 	print (pred)
# 	gene1=pred['Gene1'].tolist()
# 	gene2=pred['Gene2'].tolist()

# 	avg_scores=[]
# 	novel_genes=[]
# 	for gene in new_genes:
# 		df1=pred.loc[pred['Gene1'] == gene]
		
# 		df2=pred.loc[pred['Gene2'] == gene]
# 		df=df1.append(df2)
# 		scores=df['ypredict'].tolist()
# 		#print ('length of scores', len(scores))
# 		scores_np=np.array(scores)
# 		avg_score=np.mean(scores_np)
		
# 		avg_scores.append(avg_score)
# 		novel_genes.append(gene)

# 	print ('novel_genes', len(novel_genes), 'all_average_scores', len(avg_scores))

# 	df=pd.DataFrame({'genes': novel_genes, 'avg_scores': avg_scores})
	# df.to_csv('%s_brain_RNA_big_pool_novel_synapse_genes_avg_scores.csv'%name)
	# return df

def find_avg_scores(pred_filename, name):
	#print ('new_genes', len(new_genes))

	#pred_filename='new_all_gene_predictions.csv'
	pred=pd.read_csv(pred_filename, index_col=[0])
	print (pred)

	avg_df=pred.groupby('Gene2').mean().reset_index()
	#avg_df=avg_df.reset_index()
	avg_df.columns = ['genes', 'avg_scores']
	avg_df=avg_df.set_index('genes')
	#avg_df=avg_df.set_index('genes')
	avg_df.to_csv('%s_brain_RNA_big_pool_novel_synapse_genes_avg_scores.csv'%name)
	return avg_df


