<img src="docs/figs/logo_rc.png" width=400, padding=100>


## Recursive Algorithm for Coarse-to-fine Clusters OptimizatiON
### v 0.3.1

`raccoon` is a python 3 package for recursive clustering automatization. 
It searches for the optimal clusters in your data by running low information features removal, non-linear dimensionality reduction, and clusters identification. Tunable parameters at each of these steps are automatically set as to maximize a clustering "goodness" score. This process is then repeated recursively within each cluster identified.

This library includes

* features filtering by variance, MAD or t-SVD
* integrated unsupervised/supervised UMAP dimensionality reduction
* clusters identification by DBSCAN, HDBSCAN, SNN or Louvain
* k-NN classification
* GPU implementation with RAPIDS

Detailed documentation, API references and tutorials can be found at this [link](http://raccoon.readthedocs.org/en/latest/).

### Why recursion?

When working with complex high-dimensionality datasets, one may be interested in data relationships at different hierarchical levels. In a pet image recognition project, one may want to distinguish not only cats from dogs, but also different breeds.
While a number of hierarchical clustering methods are available, they generally tend to ignore that optimal parameters of dimensionality reduction and other steps in a typical clustering analysis are dependent on the subset of data being considered, and work instead on a single set space. 
The optimal dimensionality for separating dog breeds may lay on a different lower-dimensionality manifold than the one that allows to separate distinc species, while features that may be irrelevant in distinguishing a cat from a dog may hold considerable information at the breeds level. 
For a proper hierarchal analysis, the choice of clustering parameters should be repeated at each iteration, accounting for the new range and shape of the data subsets.
`raccoon` identifies the proper clustering parameters at each hierarchical level, by repeating the optimization recursively and independently for each identified cluster.  

### Dependencies

Beside basic scientific and plotting libraries, the current version requires

```
- scikit-learn
- scikit-network
- umap-learn
- seaborn
```

Optional dependencies include

```
- hdbscan
- rapids (see below)
```

### GPU

raccoon can be run on GPU by leveraging RAPIDS libraries. Since these libraries are still in active development, the latest versions are required to avoid issues.

```
- cupy v8.60
- cuml v0.18
- cudf v0.18
- cugraph v0.18
```

Currently there are some major (hopefully temporary) limitation in this implementation. UMAP can only run with euclidean distance, DBSCAN is the only clusters identification algorithm available.
If these do not affect your analysis, we strongly suggest to activate the GPU option, especially for larger dataset that could lead to exceptionally cumbersome runs.

### Installation

raccoon can be easily installed through python standard package managers, `pip install raccoon` or `conda install raccoon`. Alternatively, to install the latest (unreleased) version you can download it from this repository by running 
 

    git clone https://github.com/fcomitani/raccoon
    cd raccoon
    python setup.py install

### Basic usage

Given a `input` dataset in pandas-like format (samples X features), the `run` function will
automatically set up a recursive clusters search with just some basic options. 

    import raccoon as rc

    cluster_membership, tree = rc.run(input, dim=2, popcut=25,
                                     optimizer='auto', dynmesh=True,
                                     metric_clu='cosine', metric_map='cosine',
                                     outpath='./output', gpu=False)

For more details on how to customize your run, library API and tutorials, please see the documentaion.

### Citation

When using this library, please cite

> F. Comitani ... A. Shlien (in preparation)

### Contributions

This library is still a work in progress and we are striving to improve it, by adding more flexibility and increase memory and time efficiency of the code. If you would like to be part of this effort, please fork the master branch and work from there. Make sure your code passes the travis build test. 

Contributions are always welcome.
