.. Introduction documentation master file, created by
   sphinx-quickstart on Sun Oct 30 18:10:26 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: figs/logo_rc.png
	:width: 300px


Home
====

Recursive Algorithm for Coarse-to-fine Clusters OptimizatiON (raccoon) is a python 3 package for recursive clustering automatization. 
It searches for the optimal clusters in your data by running low information features removal, non-linear dimensionality reduction, and clusters identification. Tunable parameters at each of these steps are automatically set as to maximize a clustering "goodness" score.
This process is then repeated recursively within each cluster identified.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   self
   installation.rst
   releases.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Features

   clustering.rst
   classification.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API

   api.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tutorial

   tutorial_MINST.rst
