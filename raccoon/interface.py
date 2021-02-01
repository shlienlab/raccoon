"""
Interface for parallelizable functions for RACCOON (Recursive Algorithm for Coarse-to-fine Clustering OptimizatiON)
F. Comitani     @2020-2021
"""

class interface:

    """ Interface for parallelizable functions. """

    def __init__(self):
        
        self.num = None
        self.df = None
        
        self.tSVD = None
        self.UMAP = None
        self.DBSCAN = None
        self.NN = None
        self.norm = None
        self.lb = None
        self.ohe = None
        self.sis = None

        self.num = None
        self.df = None

    def decompose():
        pass

    def dimRed():
        pass

    def cluster():
        pass

    def nNeighbor():
        pass

    def labelBin():
        pass

    def oneHot():
        pass
    
    def silhouette():
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


    def getValue(self, var):
        
        """ Returns value of given variable.

        Args:
            (any): input variable.
        Returns:
            (any): value of the input variable.
        """

        return var


class interfaceCPU(interface):

    """ Interface for CPU functions. """

    def __init__(self):
       
        """ Load the required CPU libraries. """

        super().__init__()

        import numpy 
        import pandas as pd
        
        from sklearn.decomposition import TruncatedSVD as tSVD
        from umap import UMAP
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors as NN
        from sklearn.preprocessing import normalize
        from sklearn.preprocessing import LabelBinarizer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.metrics import silhouette_score

        self.num = numpy
        self.df = pd

        self.tSVD = tSVD
        self.UMAP = UMAP
        self.DBSCAN = DBSCAN
        self.NN = NN
        self.norm = normalize
        self.lb = LabelBinarizer
        self.ohe = OneHotEncoder
        self.sis = silhouette_score

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
            (obj): nearest neighbors object.
        """

        return self.NN(**kwargs)

    def labelBin(self, **kwargs):

        """ Sets up label binarizer object.

        Args:
            (dict): keyword arguments for label binarizer.
        Returns:
            (obj): label binarizer object.
        """

        return self.lb(**kwargs)

    def oneHot(self, **kwargs):

        """ Sets up one-hot encoder object.

        Args:
            (dict): keyword arguments for encoder.
        Returns:
            (obj): encoder object.
        """

        return self.ohe(**kwargs)

    def silhouette(self, points, labels, **kwargs):
        
        """ Calculates the silhouette score for a set of points
            and clusters labels.

        Args:
            points (self.df.DataFrame): points coordinates.
            labels (self.df.Series): clusters membership labels.
            (dict): keyword arguments for silhouette score (e.g. metric).
        Returns:
            (int): silhouette score on given points.
        """

        return self.sis(points, labels, **kwargs)
    
    def getValue(self, var, pandas=False):

        """ Returns value of given variable,

        Args:
            var (any): input variable.
            pandas (bool): if True, do nothing.
        Returns:
            (any): value of the input variable.
        """

        if isinstance(var,(self.df.Index,self.df.Series,self.df.DataFrame)) and not pandas:
            return var.values
        
        return var

    def set(self, var):

        """ Wrapper for python set, 
            GPU friendly..

        Args:
            (any): input variable.
        Returns:
            (set): set of the input variable.
        """

        return set(var)


class interfaceGPU(interface):

    """ Interface for GPU functions. """

    def __init__(self):
       
        """ Load the required CPU libraries. """
        
        super().__init__()

        import cupy
        import cudf
        
        from cuml.decomposition import TruncatedSVD as tSVD
        from cuml.manifold.umap import UMAP as UMAP
        from cuml import DBSCAN as DBSCAN
        from cuml.neighbors import NearestNeighbors as NN
        from cuml.experimental.preprocessing import normalize
        from cuml.preprocessing import LabelBinarizer
        from cuml.preprocessing import OneHotEncoder
        
        #silhouette score GPU not availablei in cuml 0.17a (memory issues)
        #from cuml.metrics.cluster import silhouette_score
        from sklearn.metrics import silhouette_score

        self.num = cupy
        self.df = cudf
        
        self.tSVD = tSVD
        self.UMAP = UMAP
        self.DBSCAN = DBSCAN
        self.NN = NN
        self.norm = normalize
        self.lb = LabelBinarizer
        self.ohe = OneHotEncoder
        self.sis = silhouette_score

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

    def labelBin(self, **kwargs):

        """ Sets up label binarizer object.

        Args:
            (dict): keyword arguments for label binarizer.
        Returns:
            (obj): label binarizer object.
        """

        return self.lb(**kwargs)
    
    def oneHot(self, **kwargs):

        """ Sets up one-hot encoder object.

        Args:
            (dict): keyword arguments for encoder.
        Returns:
            (obj): encoder object.
        """

        return self.ohe(**kwargs)

    def silhouette(self, points, labels, **kwargs): 
        
        """ Calculates the silhouette score for a set of points
            and clusters labels.
        
        Args:
            points (self.df.DataFrame): points coordinates.
            labels (self.df.Series): clusters membership labels.
            (dict): keyword arguments for silhouette score (e.g. metric).
        Returns:
            (int): silhouette score on given points.
        """
        
        #temporary workaround until GPU implementation is fixed
        return self.sis(self.getValue(self.getValue(points)), self.getValue(labels), **kwargs)


    def getValue(self, var, pandas=False):

        """ Returns value of given variable, 
            transferring it from GPU to CPU.

        Args:
            var (any): input variable.
            pandas (bool): if True, transform cudf to pandas.
        Returns:
            (any): value of the input variable.
        """
        
        if isinstance(var,self.df.Index):
            return var.values_host
        
        elif isinstance(var,self.df.Series):
            if pandas:
                return var.to_pandas()
            return var.values_host
        
        elif isinstance(var,self.df.DataFrame):
            if pandas:
                return var.to_pandas()
            return var.values
        
        elif isinstance(var,self.num.ndarray):
            return self.num.asnumpy(var)
        
        return var.get()

    def set(self, var):

        """ Wrapper for python set,
            GPU friendly..

        Args:
            (any): input variable.
        Returns:
            (set): set of the input variable.
        """

        return set(self.getValue(var))

if __name__ == "__main__":

    pass
