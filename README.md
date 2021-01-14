<img src="docs/figs/logo_rc.png" width=400, padding=100>


## Recursive Algorithm for Coarse-to-fine Clusters OptimizatiON
### v 0.3.0

raccoon is a python 3 package for recursive clustering automatization. 
It searches for the optimal clusters in your data by running low information features removal, non-linear dimensionality reduction, and clusters identification. Tunable parameters at each of these steps are automatically set as to maximize a clustering "goodness" score. This process is then repeated recursively within each cluster identified.

This library includes

* features filtering by variance, MAD or t-SVD
* integrated UMAP dimensionality reduction
* clusters identification by DBSCAN or HDBSCAN
* k-NN classification
* GPU implementation with RAPIDS

Detailed documentation, API references and tutorials can be found at this [link](http://raccoon.readthedocs.org/en/latest/).

### Dependencies

Beside basic scientific and plotting libraries, the current version requires

```
- scikit-learn
- umap-learn
- hdbscan
- seaborn
```

Optional dependencies include

```
- rapids (see below)
```

### GPU

raccoon can be run on GPU by leveraging RAPIDS libraries. Since these libraries are still in active development, the latest versions are required to avoid issues.

```
- cupy v8.00
- cuml v0.17
- cudf v0.17
```

### Installation

raccoon can be easily installed through python standard package managers, `pip install raccoon` or `conda install raccoon`. Alternatively, to install the latest (unreleased) version you can download it from this repository by running 
 

    git clone https://github.com/fcomitani/raccoon
    cd raccoon
    python setup.py install

### Basic usage

Given a `input` dataset in pandas-like format (samples X features), the `run` function will
automatically set up a recursive clusters search with just some basic options. 

    import raccoon as rc

    clusterMembership, tree = rc.run(input, dim=2, popcut=25,
                                     optimizer='auto', dynmesh=True,
                                     metricClu='cosine', metricMap='cosine',
                                     outpath='./output', gpu=False)

For more details on how to customize your run, library API and tutorials, please see the documentaion.

### Citation

When using this library, please cite

> F. Comitani ... A. Shlien (in preparation)

### Contributions

This library is still a work in progress and we are striving to improve it, by adding more flexibility and increase memory and time efficiency of the code. If you would like to be part of this effort, please fork the master branch and work from there. Make sure your code passes the travis build test. 

Contributions are always welcome.
