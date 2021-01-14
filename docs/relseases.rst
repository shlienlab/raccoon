
===============
Release History
===============

Version 0.3.0
==============

Features
--------

   - Features filtering can be done by variance, Median Absolute
     Deviation (MAD) or truncated Single Value Decomposition (t-SVD)
   - Integrated UMAP dimensionality reduction
   - DBSCAN and HDBSCAN are available for clusters identification
   - k-Nearest Neighbours classification
   - Parameters range selection can be automatic, for large jobs
     the mesh can be dynamically adapted 
   - A short tutorial with the MNIST dataset is available
 
   - GPU implementation with RAPIDS is now available

To Do 
=====
...
   - Currently parallelization is only available through 
     of sklearn and UMAP and is automatically active when possible. 
     Implement threading on separate sibling clusters instances
   
   - Add Shared Nearest Neihbours (SNN) to the clusters identification tools
   - Add more objective functions
   - Add weights to objective function to account for the datapoints discarded as noise
...   
    - Add automatic testing with nose/pytest
   
   - Make HDBSCAN,seaborn optional
   - Add more clustering identification options

