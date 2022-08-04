"""
Parallelizable functions interface for RACCOON
F. Comitani     @2020-2022
"""

class Interface:

    """ Interface for parallelizable functions. """

    def __init__(self):

        self.gpu        = False
        self.num        = None
        self.df         = None

        self.tSVD       = None
        self.UMAP       = None
        self.DBSCAN     = None
        self.NN         = None
        self.norm       = None
        self.lb         = None
        self.ohe        = None
        self.sis        = None
        self.pwd        = None
        self.louvain    = None

        self.num        = None
        self.df         = None

    def decompose(self):
        pass

    def dim_red(self):
        pass

    def cluster(self):
        pass
    
    def louvain(self):
        pass

    def n_neighbor(self):
        pass

    def label_bin(self):
        pass

    def one_hot(self):
        pass

    def silhouette(self):
        pass

    def dunn(self):
        pass
    
    def get_value(self, var):
        """ Returns value of given variable.

        Args:
            (any): input variable.
        Returns:
            (any): value of the input variable.
        """

        return var
    
    @staticmethod
    def filter_key(dct, keys):
        """ Remove entry from dictionary by key.

        Args:
            dct (dict): dictionary to change.
            key (obj): key or list of keys to filter.
        Returns
            (dict): filtered dictionary.
        """
        if not isinstance(keys, list):
            keys = [keys]

        return {key: value for key, value in dct.items()
                if key not in keys}


class InterfaceCPU(Interface):

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
        from sklearn.metrics import pairwise_distances as pwd
        from sknetwork.clustering import Louvain 

        self.gpu        = False

        self.num        = numpy
        self.df         = pd

        self.tSVD       = tSVD
        self.UMAP       = UMAP
        self.DBSCAN     = DBSCAN
        self.louvain    = Louvain
        self.NN         = NN
        self.norm       = normalize
        self.lb         = LabelBinarizer
        self.ohe        = OneHotEncoder
        self.sis        = silhouette_score
        self.pwd        = pwd

    def decompose(self, **kwargs):
        """ Sets up features filtering object.

        Args:
            (dict): keyword arguments for features filtering.
        Returns:
            (obj): features filtering object.
        """

        return self.tSVD(**kwargs)

    def dim_red(self, **kwargs):
        """ Sets up dimensionality reduction object.

        Args:
            (dict): keyword arguments for dimensionality reduction.
        Returns:
            (obj): dimensionality reduction object.
        """

        return self.UMAP(**kwargs)

    def cluster(self, pj, **kwargs):
        """ Sets up clusters identification object with DBSCAN.

        Args:
            pj (DataFrame): projected data to cluster.
            (dict): keyword arguments for clusters identification.
        Returns:
            (obj): clusters identification object.
        """

        clusterer = self.DBSCAN(**kwargs)
        return clusterer.fit_predict(pj)
    
    def cluster_louvain(self, pj, **kwargs):
        """ Sets up clusters identification object with Louvain..

        Args:
            pj (DataFrame): adjacency matrix to cluster.
            (dict): keyword arguments for clusters identification.
        Returns:
            (obj): clusters identification object.
        """
        
        clusterer = self.louvain(**kwargs)
        return clusterer.fit_transform(pj)

    def n_neighbor(self, **kwargs):
        """ Sets up nearest neighbors object.

        Args:
            (dict): keyword arguments for nearest neighbors.
        Returns:
            (obj): nearest neighbors object.
        """

        return self.NN(**kwargs)

    def label_bin(self, **kwargs):
        """ Sets up label binarizer object.

        Args:
            (dict): keyword arguments for label binarizer.
        Returns:
            (obj): label binarizer object.
        """

        return self.lb(**kwargs)

    def one_hot(self, **kwargs):
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

    def inv_cov(self, data):
        """ Attempts to find the inverse of the covariance matrix
            if the matrix is singular use the Moore-Penrose pseudoinverse.

        Args:
            data (self.df.Dataframe or ndarray): matrix containing the datapoints.
        Returns:
            (ndarray): the (pseudo)inverted covariance matrix.
        """

        try:
            return self.num.linalg.inv(self.num.cov(data.T))
        except BaseException:
            return self.num.linalg.pinv(self.num.cov(data.T))

    def dunn(self, points, labels, **kwargs):
        """ Calculates the dunn index score for a set of points
            and clusters labels.
            WARNING: slow!

        Args:
            points (self.df.DataFrame): points coordinates.
            labels (self.df.Series): clusters membership labels.
            (dict): keyword arguments for pairwise distances (e.g. metric).
        Returns:
            (int): dunn index on given points.
        """

        centroids = self.num.array([self.get_value(points[labels == l].mean())
                                    for l in self.num.unique(labels)])

        if kwargs['metric'] == 'mahalanobis':
            invcov = self.inv_cov(centroids)
            samples = [self.pwd(points[labels == l], **kwargs,
                VI=self.inv_cov(points[labels == l])).max()
                for l in self.num.unique(labels)]
            kwargs['VI'] = invcov
        else:
            samples = [self.pwd(points[labels == l], **kwargs).max()
                       for l in self.num.unique(labels)]

        inter = self.pwd(centroids, **kwargs)
        self.num.fill_diagonal(inter, inter.max() + 1)
        inter = self.num.amin(inter)
        intra = self.num.amax(samples)

        return inter / intra

    def get_value(self, var, pandas=False):
        """ Returns value of given variable,

        Args:
            var (any): input variable.
            pandas (bool): if True, do nothing.
        Returns:
            (any): value of the input variable.
        """

        if isinstance(var, (self.df.Index, self.df.Series,
                            self.df.DataFrame)) and not pandas:
            return var.values

        return var

    def set(self, var):
        """ Wrapper for python set,
            GPU friendly.

        Args:
            (any): input variable.
        Returns:
            (set): set of the input variable.
        """

        return set(var)


class InterfaceGPU(Interface):

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
        from cuml.metrics.pairwise_distances import pairwise_distances as pwd
        from cugraph import Graph
        from cugraph.community.louvain import louvain

        # silhouette score GPU not availablei in cuml 0.17a (memory issues)
        #from cuml.metrics.cluster import silhouette_score
        from sklearn.metrics import silhouette_score

        self.gpu        = True

        self.num        = cupy
        self.df         = cudf

        self.tSVD       = tSVD
        self.UMAP       = UMAP
        self.DBSCAN     = DBSCAN
        self.louvain    = louvain
        self.NN         = NN
        self.norm       = normalize
        self.lb         = LabelBinarizer
        self.ohe        = OneHotEncoder
        self.sis        = silhouette_score
        self.pwd        = pwd
        self.graph      = Graph


    def decompose(self, **kwargs):
        """ Sets up features filtering object.

        Args:
            (dict): keyword arguments for features filtering.
        Returns:
            (obj): features filtering object.
        """

        return self.tSVD(**kwargs)

    def dim_red(self, **kwargs):
        """ Sets up dimensionality reduction object.

        Args:
            (dict): keyword arguments for dimensionality reduction.
        Returns:
            (obj): dimensionality reduction object.
        """
        
        #typo in cuml
        if 'target_weight' in kwargs.keys():
            kwargs['target_weights']=kwargs['target_weight']
            del kwargs['target_weight']

        return self.UMAP(**self.filter_key(kwargs, 'metric'))

    def cluster(self, pj, **kwargs):
        """ Sets up clusters identification object with DBSCAN.

        Args:
            pj (DataFrame): projected data to cluster.
            (dict): keyword arguments for clusters identification.
        Returns:
            (obj): clusters identification object.
        """

        clusterer= self.DBSCAN(
            **self.filter_key(kwargs, ['metric', 'leaf_size', 'n_jobs', 'metric_params']))
        return clusterer.fit_predict(pj)

    def build_graph(self, pj):
        """ Builds a graph from an adjacency matrix 

        Args:
            pj (DataFrame): adjacency matrix to cluster.
        Returns:
            (Graph): cuGraph undirected graph
        """
        
        g=self.graph()
        g.from_numpy_matrix(self.get_value(pj))
        return g.to_undirected()

    def cluster_louvain(self, pj, **kwargs):
        """ Sets up clusters identification object with Louvain.

        Args:
            pj (Graph): cuGraph undirected graph from adjacency matrix
            (dict): keyword arguments for clusters identification.
        Returns:
            (obj): clusters identification object.
        """
        
        #Apparently there's no way to get a Graph directly from cupy adjacency matrix...
        #so I need to make sure that pj is a pd dataframe... 
        #return self.louvain(self.graph().from_pandas_adjacency(pj), **kwargs)
        #temporary super-inefficient workaround
        #also the results don't match, to be reviewed, don't use!!!
        parts, modularity = self.louvain(pj, **kwargs)
        return parts['partition']

    def n_neighbor(self, **kwargs):
        """ Sets up nearest neighbors object.

        Args:
            (dict): keyword arguments for nearest neighbors.
        Returns:
            (obj): features nearest neighbors.
        """

        return self.NN(**self.filter_key(kwargs, ['n_jobs', 'metric_params']))

    def label_bin(self, **kwargs):
        """ Sets up label binarizer object.

        Args:
            (dict): keyword arguments for label binarizer.
        Returns:
            (obj): label binarizer object.
        """

        return self.lb(**kwargs)

    def one_hot(self, **kwargs):
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

        # temporary workaround until GPU implementation is fixed
        return self.sis(
            self.get_value(
                self.get_value(points)),
            self.get_value(labels),
            **kwargs)

    def inv_cov(self, data):
        """ Attempts to find the inverse of the covariance matrix
            if the matrix is singular use the Moore-Penrose pseudoinverse.

        Args:
            data (self.df.Dataframe or ndarray): matrix containing the datapoints.
        Returns:
            (ndarray): the (pseudo)inverted covariance matrix.
        """

        try:
            return self.num.linalg.inv(self.num.cov(data.T))
        except BaseException:
            return self.num.linalg.pinv(self.num.cov(data.T))

    def dunn(self, points, labels, **kwargs):
        """ Calculates the dunn index score for a set of points
            and clusters labels.
            WARNING: slow!

        Args:
            points (self.df.DataFrame): points coordinates.
            labels (self.df.Series): clusters membership labels.
            (dict): keyword arguments for pairwise distances (e.g. metric).
        Returns:
            (int): dunn index on given points.
        """

        # TODO: clean up all this back and forth between gpu and cpu

        inter = self.pwd(self.num.array([self.get_value(points[labels == l].mean())
            for l in self.num.unique(labels).get()]), **kwargs)
        self.num.fill_diagonal(inter, inter.max() + 1)

        inter = self.num.amin(inter)

        intra = self.num.amax(self.num.array([self.pwd(
            points[labels == l], **kwargs).max().max()
            for l in self.num.unique(labels).get()]))

        return self.get_value(inter / intra)

    def get_value(self, var, pandas=False):
        """ Returns value of given variable,
            transferring it from GPU to CPU.

        Args:
            var (any): input variable.
            pandas (bool): if True, transform cudf to pandas.
        Returns:
            (any): value of the input variable.
        """

        if isinstance(var, self.df.Index):
            return var.values_host

        elif isinstance(var, self.df.Series):
            if pandas:
                return var.to_pandas()
            return var.values_host

        elif isinstance(var, self.df.DataFrame):
            if pandas:
                return var.to_pandas()
            return var.values

        elif isinstance(var, self.num.ndarray):
            return self.num.asnumpy(var)

        elif isinstance(var, list) or isinstance(var, float)\
            or isinstance(var, int):
            return var

        return var.get()

    def set(self, var):
        """ Wrapper for python set,
            GPU friendly..

        Args:
            (any): input variable.
        Returns:
            (set): set of the input variable.
        """

        return set(self.get_value(var))


if __name__ == "__main__":

    pass
