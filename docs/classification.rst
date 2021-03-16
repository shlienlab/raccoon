
====================
Classification
====================

raccoon provides an implementation of a 
basic distance-weighted k-nearest neighbours classifier, adapted to
take as input the maps trained with our recursive clustering. 

Each input datapoint follows the same preprocessing steps as 
the original dataset and it's projected onto the embedded space 
at different levels of the hierarchy.
Clusters assignment is calculated by averaging the nearest neighbours 
classes and weighting them as a function of their distance.

To run this classifier, :code:`savemap` must be active during the clustering.

To k-NN object, has to be initialised with the dataset do be predicted,
the original dataset used to build the clusters, their membership
table (as output by :code:`recursive_clustering`) 
and the path to the reference folder (:code:`raccoon_data`) 
containing the trained maps. It also take an output folder for logging purposes
and a debugging mode switch.

.. code-block:: python
  
  from raccoon.utils.classification import KNN

  obj = KNN(df_to_predict, df, cluster_membership, refpath=r'./raccoon_data', outpath=r'./')
  obj.assign_membership()

  output = obj.membership
  
The output is in the same one-hot-encoded matrix
(rows as samples, columns classes) as the recursive clustering assignment.


