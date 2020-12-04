"""
Interface for parallelizable functions for RACCOON (Recursive Algorithm for Coarse-to-fine Clustering OptimizatiON)
F. Comitani     @2020
"""

class interface:

    """ Interface for parallelizable functions. """

    def __init__(self):
        
        self.tSVD = None
        self.UMAP = None
        self.DBSCAN = None
        self.NN = None

    def decompose():
        pass

    def dimRed():
        pass

    def cluster():
        pass

    def nNeighbor():
        pass

    def filterKey(self, dct, keys):

        """ Remove entry from dictionary by key.

        Args:
            dct (dict): dictionary to change.
            key (obj): key or list of keys to filter.
        Returns
            (dict): filtered dictionary.
        """
        if not isinstance(keys, list):
            keys=[keys]

        return {key:value for key,value in dct.items()
              if key not in keys}


class interfaceCPU(interface):

    """ Interface for CPU functions. """

    def __init__(self):
       
        """ Load the required CPU libraries. """

        super().__init__()

        from sklearn.decomposition import TruncatedSVD as tSVD
        from umap import UMAP
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors as NN

        self.tSVD = tSVD
        self.UMAP = UMAP
        self.DBSCAN = DBSCAN
        self.NN = NN

    def decompose(self, **kwargs):
        
        """ Sets up features filtering object.

        Args: 
            (dict): keyword arguments for features filtering.
        Returns:
            (obj): features filtering object.
        """

        return self.tSVD(**kwargs)

    def dimRed(self, **kwargs):

        """ Sets up dimensionality reduction object.

        Args:
            (dict): keyword arguments for dimensionality reduction.
        Returns:
            (obj): dimensionality reduction object.
        """

        return self.UMAP(**kwargs)

    def cluster(self, **kwargs):

        """ Sets up clusters identification object.

        Args:
            (dict): keyword arguments for clusters identification.
        Returns:
            (obj): clusters identification object.
        """

        return self.DBSCAN(**kwargs)

    def nNeighbor(self, **kwargs):

        """ Sets up nearest neighbors object.

        Args:
            (dict): keyword arguments for nearest neighbors.
        Returns:
            (obj): features nearest neighbors.
        """

        return self.NN(**kwargs)



class interfaceGPU(interface):

    """ Interface for GPU functions. """

    def __init__(self):
       
        """ Load the required CPU libraries. """
        
        super().__init__()

        from cuml.decomposition import TruncatedSVD as tSVD
        from cuml.manifold.umap import UMAP as UMAP
        from cuml import DBSCAN as DBSCAN
        from cuml.neighbors import NearestNeighbors as NN
        #import cudf
        #import cupy
        
        self.tSVD = tSVD
        self.UMAP = UMAP
        self.DBSCAN = DBSCAN
        self.NN = NN
        
    def decompose(self, **kwargs):
        
        """ Sets up features filtering object.

        Args: 
            (dict): keyword arguments for features filtering.
        Returns:
            (obj): features filtering object.
        """

        return self.tSVD(**kwargs)

    def dimRed(self, **kwargs):

        """ Sets up dimensionality reduction object.

        Args:
            (dict): keyword arguments for dimensionality reduction.
        Returns:
            (obj): dimensionality reduction object.
        """
        
        return self.UMAP(**self.filterKey(kwargs, 'metric'))

    def cluster(self, **kwargs):

        """ Sets up clusters identification object.

        Args:
            (dict): keyword arguments for clusters identification.
        Returns:
            (obj): clusters identification object.
        """

        return self.DBSCAN(**self.filterKey(kwargs, ['metric','leaf_size','n_jobs']))

    def nNeighbor(self, **kwargs):

        """ Sets up nearest neighbors object.

        Args:
            (dict): keyword arguments for nearest neighbors.
        Returns:
            (obj): features nearest neighbors.
        """

        return self.NN(**self.filterKey(kwargs, 'n_jobs'))

if __name__ == "__main__":

    pass
