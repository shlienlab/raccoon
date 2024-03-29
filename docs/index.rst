.. image:: figs/logo_rc.png
	:width: 400px

|
Welcome to RACCOON's documentation!
===================================

Resolution-Adaptive Coarse-to-fine Clusters OptimizatiON (RACCOON) is a Python 3 package for top-down clustering automatization. 
It searches for the optimal clusters in your data by running low information features removal, non-linear dimensionality reduction, and clusters identification. Tunable parameters at each of these steps are automatically set as to maximize a clustering "goodness" score.
This process is then repeated iteratively within each cluster identified.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   self
   installation.rst
   releases.rst
   faq.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Features

   clustering.rst
   classification.rst
   update.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tutorial

   tutorial_MNIST.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API

   api.rst
