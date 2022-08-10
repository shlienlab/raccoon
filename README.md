<img src="docs/figs/logo_rc.png" width=400, padding=100>


## Resolution-Adaptive Coarse-to-fine Clusters OptimizatiON
### v 0.5.1

[![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/shlienlab/raccoon)](https://github.com/shlienlab/raccoon/releases/tag/v0.5.1)
[![PyPI](https://img.shields.io/pypi/v/raccoon-cluster)](https://pypi.org/project/raccoon-cluster/)
[![Licence](https://img.shields.io/github/license/shlienlab/raccoon)](https://github.com/shlienlab/raccoon/blob/main/LICENSE)
[![GitHub top language](https://img.shields.io/github/languages/top/shlienlab/raccoon)](https://github.com/shlienlab/raccoon/search?l=python)
[![Documentation Status](https://readthedocs.org/projects/raccoon-cluster/badge/?version=latest)](https://raccoon-cluster.readthedocs.io/en/latest/?badge=latest)


RACCOON (`raccoon`) is a Python 3 package for top-down clustering automatization. 
It searches for the optimal clusters in your data by running low information features removal, non-linear dimensionality reduction, and clusters identification. Tunable parameters at each of these steps are automatically set to maximize a clustering "goodness" score. This process is then repeated iteratively within each cluster identified.

This library includes

* features filtering by variance, MAD or t-SVD
* integrated unsupervised/supervised UMAP non-linear dimensionality reduction
* clusters identification by DBSCAN, HDBSCAN, SNN or Louvain
* optimization with grid search, differential evolution or TPE
* k-NN classification
* GPU implementation with RAPIDS

Detailed documentation, API references, FAQ and tutorials can be found at this [link](https://raccoon-cluster.readthedocs.io/en/latest/).

### Dependencies

Besides basic scientific and plotting libraries, the current version requires

```
- scikit-learn
- scikit-network
- umap-learn
- optuna
- seaborn
```

Optional dependencies include

```
- hdbscan
- feather-format
- rapids (see below)
```

### GPU

raccoon can be run on GPU by leveraging RAPIDS libraries. Since these libraries are still in active development, the latest versions are required to avoid issues.

```
- cupy      v8.60
- cuml      v0.18
- cudf      v0.18
- cugraph   v0.18
```

Currently, there are some limitations in this implementation (e.g. UMAP can only run with euclidean distance).
If these do not affect your analysis, we suggest activating the GPU option, especially for larger datasets that could lead to cumbersome runs. Alternatively, this option should be used for exploratory runs only.

**Important note**: the GPU implementation is still a work in progress and may change considerably in the coming versions. Please report any bug or issue you experience. 

### Scripts

Useful scripts can be found in the `scripts` folder. These include files to read hdf5 storing the output pandas dataframe in R.
See the documentation for more details. 

### Installation

Raccoon releases can be easily installed through the python standard package manager  
`pip install raccoon-cluster`.

To install the latest (unreleased) version you can download it from this repository by running 
 
    git clone https://github.com/shlienlab/raccoon
    cd raccoon
    python setup.py install

### Basic usage

Given an `input` dataset in pandas-like format (samples X features), the `run` function will
automatically set up a clusters search with just some basic options. 

    import raccoon as rc

    cluster_membership, tree = rc.cluster(input, dim=2, pop_cut=25,
                                     optimizer='auto', dyn_mesh=True,
                                     metric_clu='cosine', metric_map='cosine',
                                     save_map=True, chk=True,
                                     out_path='./output', gpu=False)

### Citation

When using this library, please cite

> F. Comitani, J. O. Nash, S. Cohen-Gogo, A. Chang, T. T. Wen, A. Maheshwari, B. Goyal, E. S. L. Tio, K. Tabatabaei, R. Zhao, L. Brunga, J. E. G. Lawrence, P. Balogh, A. Flanagan, S. Teichmann, B. Ho, A. Huang, V. Ramaswamy, J. Hitzler, J. Wasserman, R. A. Gladdy, B. C. Dickson, U. Tabori, M. J. Cowley, S. Behjati, D. Malkin, A. Villani, M. S. Irwin and A. Shlien, "Multi-scale transcriptional clustering and heterogeneity analysis reveal diagnostic classes of childhood cancer" (under review).


### Contributions

This library is still a work in progress and we are striving to improve it, by adding more flexibility and increase the memory and time efficiency of the code. If you would like to be part of this effort, please fork the main branch and work from there. We plan on adding pytests and CI in the coming versions. 

Contributions are always welcome.
