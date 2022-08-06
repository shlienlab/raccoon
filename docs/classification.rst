
==============
Classification
==============

raccoon provides an implementation of a 
basic distance-weighted k-nearest neighbours classifier adapted to
take as input the maps trained with our scale-adaptive clustering. 

Each input datapoint follows the same preprocessing steps as 
the original dataset and it's projected onto the embedded space 
at different levels of the hierarchy.
Clusters assignment is calculated by averaging the nearest neighbours 
classes and weighting them as a function of their distance.

To run this classifier, :code:`save_map` must be active during the clustering 
step.

A wrapper function is available 

.. code-block:: python
  
  import aroughcun as rc

  projected_membership  = rc.classify(df_to_predict, original_df, cluster_membership, 
                                      refpath=r'./raccoon_data', out_path=r'./')

Alternatively, the k-NN object can be initialized and the classification can be
called directly.

.. code-block:: python
  
  from aroughcun.utils.classification import KNN

  obj = KNN(df_to_predict, original_df, cluster_membership, 
            refpath=r'./raccoon_data', out_path=r'./',
            debug=False, gpu=False)
  obj.assign_membership()

  output = obj.membership
  
The k-NN run requires the dataset to be predicted,
the original dataset used to build the clusters, their membership
table (as output by :code:`cluster`) 
and the path to the reference folder (:code:`raccoon_data`) 
containing the trained maps. It also takes an output folder for logging purposes, a debugging mode switch and a GPU switch.

The output is in the same one-hot-encoded matrix format
(rows as samples, columns classes) as the iterative clustering output table.
