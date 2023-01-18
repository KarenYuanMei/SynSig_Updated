# SynSig_Updated: Synaptic Signatures

The synapse is a complex protein-dense structure critical for proper brain functioning. The molecular composition of the synaptic network is incompletely defined, impeding our understanding of healthy and diseased neurological functions. To address this gap, we devised a machine learning system to capture core features of the synapse from their genomic, transcriptomic, and structural patterns – a “synaptic signature” – leading to the identification of novel synaptic proteins.

Manuscript in preparation: Mei et al., "Identifying Synapse Genes Using Global Molecular Signatures."

## Dependencies:

### Batch installation of packages:
  -Packages necessary are listed in env_packages folder
  -yml, txt, and piplock versions are all included for easy installation
 
### Manual installation of dependencies:
  - use Python=3.7

  conda create --name myenv python=3.7
  
  - Numpy, Scipy, Networkx
  
  conda install -y pandas numpy scipy networkx=1.11
  
  -openpyxl
  
  conda install openpyxl
  
  -xlrd
  
  conda install xlrd

  - igraph
  
  conda install -y -c conda-forge python-igraph
  
  conda install -y libiconv # Needed for igraph to run properly
  
  - tulip-python, ndex-dev
  
  pip install tulip-python
  
  pip install ndex-dev
  
  - ddot
  
  pip install /path to ddot repo/
  
  - scikit-learn
  
  pip install -U scikit-learn
  
  -mygene
  
  pip install mygene
  
  -matplotlib
  
  pip install matplotlib
  
  -venn3
  
  pip install matplotlib_venn
  
  -nbformat
  
  pip install nbformat
  
  -seaborn 
  
  pip install seaborn

## For analysis of the datasets, some graphs were generated using R in a separate environment
## To load dependencies for the R environment:
### create R environment in jupyter:

conda activate <R-ENV>
conda install r-irkernel
Pip install Jupyter
Rscript -e 'IRkernel::installspec(name="<R-ENV>", displayname="R <R-ENV>")'

then if you start a jupyter job you can use that R environment in a notebook like any python environment

### dependencies in R environment: 
  install.packages("VennDiagram")
  install.packages("grid")
  install.packages("futile.logger")
  
 
