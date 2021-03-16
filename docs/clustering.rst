
====================
Clustering
====================


A wrapper function (:code:`run`) is available for the user's convenicence,
it takes all arguments available to the clustering object and will output the
membership assignment.

To have more control on the algorithm, one can initialize the clustering object
and call the recursion by themeslves.

.. code-block:: python

  obj = RecursiveClustering(data, **kwargs)
  obj.recurse()
  
:code:`RecursiveClustering` takes an initial matrix-like object 
(rows as samples, columns as features), and a number of optional flags,
including a matching list of labels 
for plotting (:code:`labs`).

It contains information on the chosen clustering options
and a one-hot-encoded pandas dataframe containing structure
and assignment of the hierarchy of identified classes.  

.. code-block:: python

  obj.clus_opt

The numerous flags available to :code:`RecursiveClustering` 
allows for great flexibility in the clustering analysis.

Among these we find

================  ================================================================= 
:code:`maxdepth`  sets the maximum depth at which the clustering is allowed to go,
                  if :code:`None` (default) the recursion will proceed until other 
                  conditions for its termination are met.
:code:`savemap`   if active, trained maps, preprocessing status and feature filters 
                  will be saved to disk
:code:`outpath`   set the path to a directory where all the plots and output data 
                  will be stored
================  =================================================================

For the full list of available options and their use, see api_.


Normalization
==============

As a first step in the clustering analysis, the data can by activating the
:code:`norm` flag in the recursive clustering object. It takes default :code:`sklearn`
norms, and is set by default at :code:`'l2'`. If :code:`None`, no normalization will be applied prior
to the features removal step.


Optimizers
==========

There are currently two available optimizers, to be set with the :code:`optimizer` flag,
and the option to let the algorithm decide which to use on its own.

===============  ============================================================  
:code:`'grid'`   **Grid Search**: given a set of parameters ranges and steps, 
                 the tool will evaluate all possible combinations
                 on the defined mesh
:code:`'de'`     **Differential Evolution**: a simple evolutionary algorithm,
                 it requires to set a number of candidates for the parametric 
                 space search and a number of maximum iterations [Storn1997]_
:code:`'auto'`   **Automatic**: dynamically adapt the optimizer choice 
                 to the dataset size. Grid Search will be used for larger
                 coarse searches, DE will be saved for smaller datasets and 
                 more detailed runs.
===============  ============================================================

While Grid Search requires to define the exact set of points to explore, either directly
or indirectly by setting the explorable ranges and steps for each parameter (see the next sections), 
Differential Evolution requires a number of candidate points to be set with :code:`depop`
and maximum iterations :code:`deiter` which will stop the optimization unless a solutions 
improvment tolerance limit (set by default) is hit. 

*Warning*: These two flags will override the parameters selection flags specific of each of the 
downstream steps. It will only keep information on their ranges (if set), to define the boundaries.

While Differential Evolution is in principle more efficient than Grid Search, it is not guaranteed
to find the absolute minimum, and the results heavily depend on how extensive the search is.
For exploratory runs, with a low number of search points, Grid Search is preferred. Differential Evolution
should be the default choice when resources for a detailed search are available.

Dynamic Mesh
------------

The recursion can easily become cumbersome with large datasets.
Decreasing the number of explored points during the optimization, however, may hinder
the convergence towards a point of absolute optimality and lead to the loss
of important details.
For this reason, we implemented a dynamic mesh, which automatically attempts to adapt
the granularity of the search at each iteration. 
By setting :code:`dynmesh` as :code:`True`, the number of grid points (or candidates
and iterations in the case of DE) are calculated as a function of the population of the 
investigated cluster.

*Warning*: Activating this function will override the chosen step or candidates/iteration numbers.

Objective Function
------------------

To drive the optimization, different clustering evaluation scores can be used

====================  ============================================================  
:code:`'silhouette'`  **Silhouette Score**: mesures the ratio of the intra-cluster 
                      and inter-cluster distances, it ranges between 1 for an optimal
                      separation and -1 for ill-defined clusters. [Rousseeuw1987]_
:code:`'dunn'`        **Dunn Index**: mesures the ratio of the intra-cluster and 
                      inter-cluster distances, it ranges between +inf for an optimal
                      separation and 0 for ill-defined clusters. [Dunn1973]_
====================  ============================================================

These scores require to measure the distances between points and/or clusters
in the embedded space. With :code:`metric_clu`, one can select which metric to use.
Standard measures, as implemented in :code:`sklearn` 
(e.g. :code:`'euclidean'`, :code:`'cosine'`, or :code:`'mahalanobis'`) are available.

When :code:`'silhouette'` is selected, its mean value on all data points is maximized. 
To assure quality in the library's outputs, set of parameters
generating a negative score are automatically discarded.

Population Cutoff
-----------------

The recursion will be terminated under two conditions: if the optimization found a 
single cluster as the best outcome or if the population lower bound is met.
The latter can be set with :code:`popcut` and should be kept as high as possible as to avoid
atomization of the clusters, but low enough to allow for the identification of subclasses.
(depending on the dataset, the suggested values are between 10 and 50).

Low-information Filtering
=========================

These are the methods currently available for low-information features filtering:

===================  ========================================================================== 
:code:`'variance'`   **Variance** : after ordering them by variance, remove the features
                     that fall below a cut-off percentage of cumulative variance
:code:`'MAD'`        **Median Absolute Deviation**: like :code:`variance` but with MAD instead 
:code:`'t-SVD'`      **Truncated Single Value Decomposition**: applies t-SVD to the data, 
                     requires to set the number of output components [Hansen1987]_ 
===================  ==========================================================================  

Although this step is not strictly necessary to run UMAP, it can considerably improve the outcome
of the clustering, by removing noise and batch effects emerging in the low information features.

All these methods are set with :code:`filterfeat` and require a cutoff, a percentage of the cumulative variance/MAD to be removed, or 
the number of output components in t-SVD. 
This is a tunable parameter and is part of the optimization process, its range and step
can be set with :code:`ffrange` and :code:`ffpoints` respectively.

For example, setting 

.. code-block:: python

  filterfeat = 'MAD'
  ffrange = 'logspace'
  ffpoints = 25

will run the optimization on a logarithmic space between .3 and .9 in cumulative
MAD with 25 mesh points.

Dimensionality Reduction
========================

Following the low-information features removal is the dimensionality reduction by means of UMAP.
Here there are a number of flags that one could set, mostly inherited by UMAP itself, the
most important being :code:`dim`, the dimensionality of the target space.
One should take particular care in chosing this number, as it can affect both
the results and the efficiency of the algorithm. The choice of metric for the objective 
function will also depend on this value, as :code:`'euclidean'` distances are only viable in two 
dimensions.

We suggest you leave the choice of mapping metric (:code:`metric_map`), the number of epochs (:code:`epochs`) 
and learning rate (:code:`lr`), to their default values unless you know what you are doing.

Finally, as in the case of the features removal step, the number of nearest neighbours,
which defines the scale at which the dimensionality reduction is performed, is left as tunable
by the optimizer. You can chose range and number of points (if Grid Search is active) with
:code:`neirange` and :code:`neipoints` respectively.
If the range is left to be guessed automatically, for example as a logarithmic
space based on the population (:code:`'logspace'`), a factor can be set to reduce the 
value proportionally (:code:`neifactor`) in the presence of particularly large datasets,
as high values of this parameters can impact the performance considerably.


Clusters Identification
=======================

The clusters identification tool is chosen with the :code:`clusterer` flag

=================  ================================================================  
:code:`'DBSCAN'`   **Density-Based Spatial Clustering of Applications with Noise**: 
                   density based clustering, requires an :math:`$\epsilon$` distance 
                   to define clusters neighborhood [Ester1996]_
:code:`'HDBSCAN'`  **Hierarchical DBSCAN**: based on DBSCAN, it attempts to remove
                   the dependency on :math:`$\epsilon$` but is still affected by the 
                   choice of minimum cluster population [Campello2013]_
=================  ================================================================

Depending on which method has been chosen, different parameters are set as tunable for 
the optimizer (e.g. :math:`$\epsilon$` for DBSCAN or minimum population for HDBSCAN).
By means of :code:`cparmrange` one can set the range to be explored. By default this is set
as :code:`guess` which allows the algorithm to find an ideal range based on the elbow method.

If :code:`'DBSCAN'` is chosen as clusterer, its minimum value of cluster size can also be set
with :code:`minclusize`.

This step is also affected by the choice of :code:`metric_clu` as distances need to be measured
in the embedded space.

For those clustering algorithm that allow to discard points at noise, the :code:`outliers`
flag allows the user to chose what to do with these points:

==================  ================================================================  
:code:`'ignore`     points marked as noise will be left as such and discared at the 
                    next iteration.
:code:`'reassign'`  attempts to force the assignment of a cluster membership to all 
                    the points marked as noise by means ofnearest neighbours.
==================  ================================================================


Given that this step is in most cases considerably less expensive than the other two, 
and that the DE algorithm efficacy is considerably reduced above 2 dimensions, the 
search for this parameter is set by default as a Grid Search with fine mesh.


Transform-only data
===================

Occasionally you may want to train your clusters only on a subset of the data, while still
use them to classify some held-out set.

By setting :code:`transform` you can ask the algorithm to run each one of the clustering steps
recursively only on a given subset, while still forcing the membership assignment by means of k-NN 
to the rest of the data.

The full dataset has to be given as input, including the data to project, but not used in the training.
:code:`transform` takes a list-like object containing the indices of the points *not* to be used
for the training.

Activating this function will produce extra plots at each iteration, of projection maps 
color coded according to which points were used for the training and which transformed only.


Saving hierarchy information
============================

The resulting clustering membership will be stored as a one-hot-encoded pandas dataframe in the :code:`obj.clus_opt` variable.
However, auxiliary functions are available to store the hierarchy information as an :code:`anytree` object as well.

.. code-block:: python
  
  import raccoon.utils.trees as trees

  tree = trees.build_tree(obj.clus_opt)

:code:`build_tree` requires the membership assignment table as input and optionally a path to where to save the tree in :code:`json` format.
By default it will be saved in the home directory of the run.
To load a tree from the :code:`json` foile :code:`load_tree` only requires its path.

Repeating a run
===============

The :code:`fromfile` flag takes the path to a :code:`paramdata.csv` as input and allows the user to repeat a run using the optimal parameters and skipping the search
altogether. Activating this flag will override all other parameters. This can be useful to reproduce past works, save more files (e.g. trained maps) or plots. 

GPU
===

If a GPU is available on your system you can speed up your calculations by activating the `gpu` boolean flag when initializing the run.
You will need `CuPy <https://cupy.dev/>`_ and `RAPIDS <https://rapids.ai/>`_ to be installed.
See :code:`requirements.txt` for details on modules and versions. 

References
----------
        
.. [Storn1997] Storn R. and Price K. (1997),  "Differential Evolution - a Simple and Efficient Heuristic for Global Optimization over Continuous Spaces", Journal of Global Optimization, 11: 341-359.
.. [Rousseeuw1987] Rousseeuw P. J. (1987), "Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis", Computational and Applied Mathematics, 20: 53-65.
.. [Dunn1973] Dunn J. C. (1973), "Well-Separated Clusters and Optimal Fuzzy Partitions", Journal of Cybernetics, 4:1, 95-104.
.. [Hansen1987] Hansen, P. C. (1987), "The truncatedSVD as a method for regularization", BIT, 27:,: 534–553. 
.. [Ester1996] Ester M., Kriegel H. P., Sander J. and Xu X. (1996), “A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise”, Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining, 226-231.
.. [Campello2013] Campello R.J.G.B., Moulavi D., Sander J. (2013) Density-Based Clustering Based on Hierarchical Density Estimates, Advances in Knowledge Discovery and Data Mining, PAKDD  Lecture Notes in Computer Science, vol 7819.
