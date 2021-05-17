
===============
Release History
===============

Version 0.4
===============

Features
--------

   - Features filtering can be done by variance, Median Absolute
     Deviation (MAD) or truncated Single Value Decomposition (t-SVD)
   - Integrated UMAP dimensionality reduction
   - Supervised dimensionality reduction with UMAP
   - DBSCAN, HDBSCAN, Shared Nearest Neighbours and Louvain community
        detection  are available for clusters identification
   - k-Nearest Neighbours classification
   - Parameters range selection can be automatic, for large jobs
     the mesh can be dynamically adapted 
   - Old runs can be resumed or extended for a finer search 
   - A short tutorial with the MNIST dataset is available


   - GPU implementation with RAPIDS

...

To Do 
=====
   - Currently parallelization is only available through 
     of sklearn and UMAP and is automatically active when possible. 
     Implement threading on separate sibling clusters instances
   - Add weights to objective function to account for the datapoints discarded as noise
...   

