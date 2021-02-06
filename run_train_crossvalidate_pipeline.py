#Goal: 1) define training genes and GO scores
#	   2) five-fold cross-validation


import pandas as pd
import numpy as np
import csv
import random

import ddot
from ddot import Ontology

from itertools import combinations, combinations_with_replacement
from itertools import product
from collections import defaultdict

from define_gene_objects import define_features, Gene, PairOfGenes, find_input_features, load_feature, create_feature_value_dict, get_feature_value, create_GO_score_dict, create_gene_list, find_pos_genes_in_training, find_gene_objects, find_feature_array, create_input_pair_objects, run_adaboost, run_svm_regressor, run_random_forest, find_data_genes, create_data_gene_list, run_new_rf, find_data_array, find_avg_scores
from load_data_functions import get_gene_names
from find_training_genes_scores_functions import random_select, find_pos_neg_input, divide_5fold, find_pos_neg_chunks, define_training_test, find_GO_ont, find_GO_score_matrix, find_input_gene_GO_scores


syngo_file='../correct_db/corr_syngo_cc.csv'
index_file='../../SynSig/synsig_random_forest/big_pool_genes_index.csv'

big_pool=get_gene_names(index_file)
feature_value_dict = create_feature_value_dict(big_pool)
print ("DONE1")