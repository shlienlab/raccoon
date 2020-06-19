<img src="docs/figs/logo_rc.png" width=400, padding=100>


## Recursive Algorithm for Coarse-to-fine Clusters OptimizatiON

raccoon is a python 3 package for recursive clustering automatization. 
It searches for the optimal clusters in your data by running low information features removal, non-linear dimensionality reduction, and clusters identification. Tunable parameters at each of these steps are automatically set as to maximize a clustering "goodness" score. This process is then repeated recursively within each cluster identified.

This library includes

* features filtering by variance, MAD or t-SVD
* integrated UMAP dimensionality reduction
* clusters identification by DBSCAN or HDBSCAN
* k-NN classification

Detailed documentation, API references and tutorials can be found at this [link](http://raccoon.readthedocs.org/en/latest/).

### Dependencies

Beside basic scientific and plotting libraries, the current version requires

```
- scikit-learn
- umap-learn
- hdbscan
```

<!-- To run the tests, you also must have `nose` installed. -->

### Installation

raccoon can be easily installed through python standard package managers, `pip install raccoon` or `conda install raccoon`. Alternatively, to install the latest (unreleased) version you can download it from this repository by running 
 

    git clone https://github.com/fcomitani/raccoon
    cd raccoon
    python setup.py install

### Citation

When using this library, please cite

> F. Comitani ... A. Shlien (in preparation)

### Contributions

This library is still a work in progress and we are striving to improve it, by adding more flexibility and increase memory and time efficiency of the code. If you would like to be part of this effort, please fork the master branch and work from there. Make sure your code passes the travis build test. We will soon add tests to be run with `nose`. 

Contributions are always welcome.


<!--
If you would like to contribute a feature then fork the master branch (fork the release if you are fixing a bug). Be sure to run the tests before changing any code. You'll need to have [nosetests](https://github.com/nose-devs/nose) installed. The following command will run all the tests:

```
python setup.py test
```

Let us know what you want to do just in case we're already working on an implementation of something similar. This way we can avoid any needless duplication of effort. Also, please don't forget to add tests for any new functions.

-->