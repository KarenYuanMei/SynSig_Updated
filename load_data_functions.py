#Goal: to load different files into pandas

import pandas as pd

def get_gene_names(filename):
	df=pd.read_csv(filename)
	genes=df['genes'].tolist()
	return genes