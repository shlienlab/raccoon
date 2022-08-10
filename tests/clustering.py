"""
Set of standardized tests for the clustering function of RACCOON
F. Comitani     @2020-2022
"""
import raccoon

def grid_test(data, labels=None, gpu=False):
    """ Clustering test, euclidean grid, with checkpoints.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.
    """

    cluster_membership, tree = raccoon.cluster(data, lab=labels, dim=2, pop_cut=20, max_depth=3, 
                                     filter_feat='variance', optimizer='grid', metric_clu='euclidean', metric_map='cosine',
                                     dyn_mesh=True, max_mesh=3, min_mesh=3, chk=True,
                                     out_path='./out_test_grid', save_map=True, debug=True, gpu=gpu)

def mahalanobis_test(data, labels=None, gpu=False):
    """ Clustering test, mahalanobis grid.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.
    """

    cluster_membership, tree = raccoon.cluster(data, lab=labels, dim=2, pop_cut=20, max_depth=1,
                                     filter_feat='variance', optimizer='grid', metric_clu='mahalanobis', metric_map='cosine',
                                     dyn_mesh=True, max_mesh=3, min_mesh=3,
                                     out_path='./out_test_mahalanobis', save_map=True, debug=True, gpu=gpu)

def snn_test(data, labels=None, gpu=False):
    """ Clustering test, euclidean grid, SNN DBSCAN.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.
    """

    cluster_membership, tree = raccoon.cluster(data, lab=labels, dim=2, pop_cut=20, max_depth=1, clu_algo='SNN',
                                     filter_feat='variance', optimizer='grid', metric_clu='euclidean', metric_map='cosine',
                                     dyn_mesh=True, max_mesh=3, min_mesh=3,
                                     out_path='./out_test_snn', save_map=True, debug=True, gpu=gpu)

def louvain_test(data, labels=None, gpu=False):
    """ Clustering test, euclidean grid, SNN Louvain.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.
    """

    cluster_membership, tree = raccoon.cluster(data, lab=labels, dim=2, pop_cut=20, max_depth=1, clu_algo='louvain',
                                     filter_feat='variance', optimizer='grid', metric_clu='euclidean', metric_map='cosine',
                                     dyn_mesh=True, max_mesh=3, min_mesh=3,
                                     out_path='./out_test_louvain', save_map=True, debug=True, gpu=gpu)


def neifun_test(data, labels=None, gpu=False):
    """ Clustering test, custom neighbors selection function.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.
    """

    from math import sqrt


    def range_sqrt(n):

      sq = sqrt(n)
      return [sq/2, sq, sq+sq/2]

    cluster_membership, tree = raccoon.cluster(data, lab=labels, dim=2, pop_cut=20, max_depth=2,
                                     filter_feat='variance', optimizer='grid', metric_clu='euclidean', metric_map='cosine', 
				     nei_range=range_sqrt, 
                                     dyn_mesh=True, max_mesh=3, min_mesh=3,
                                     out_path='./out_test_neifun', save_map=True, debug=True, gpu=gpu)

def resume_test(data, resume_path, labels=None, gpu=False):
    """ Resume clustering test, euclidean grid loading parameters data from file.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            resume_path (string): path to parameters data file to load.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.
    """

    cluster_membership, tree = raccoon.resume(data, lab=labels, dim=2, pop_cut=15, max_depth=4,
                                     filter_feat='variance', optimizer='grid', metric_clu='euclidean', metric_map='cosine',
                                     dyn_mesh=True, max_mesh=3, min_mesh=3, refpath=resume_path, 
                                     out_path='./out_test_resume', save_map=True, debug=True, gpu=gpu)


def de_test(data, labels=None, gpu=False):
    """ Clustering test, euclidean Differential Evolution.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.    
    """

    cluster_membership, tree = raccoon.cluster(data, lab=labels, dim=2, pop_cut=5, max_depth=2,
                                     filter_feat='variance', optimizer='de', 
                                     search_candid=5, search_iter=5,
                                     metric_clu='euclidean', metric_map='cosine',
                                     dyn_mesh=True, max_mesh=4, min_mesh=4,
                                     out_path='./out_test_de', save_map=True, debug=True, gpu=gpu)


def tpe_test(data, labels=None, gpu=False):
    """ Clustering test, euclidean TPE.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.    
    """

    cluster_membership, tree = raccoon.cluster(data, lab=labels, dim=2, pop_cut=5, max_depth=2,
                                     filter_feat='variance', optimizer='tpe', 
                                     search_candid=25, tpe_patience=10,
			             metric_clu='euclidean', metric_map='cosine',
                                     dyn_mesh=False, clu_algo='DBSCAN',
                                     out_path='./out_test_tpe', save_map=True, debug=True, gpu=gpu)

def auto_test(data, labels=None, gpu=False):
    """ Clustering test, euclidean automatic selection.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.    
    """

    cluster_membership, tree = raccoon.cluster(data, lab=labels, dim=2, pop_cut=20, max_depth=1,
                                     filter_feat='variance', optimizer='auto', metric_clu='euclidean', metric_map='cosine',
                                     dyn_mesh=True, max_mesh=3, min_mesh=3,
                                     out_path='./out_test_auto', save_map=True, debug=True, gpu=gpu)


def tsvd_test(data, labels=None, gpu=False):
    """ Clustering test, euclidean grid with t-SVD.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.    
    """

    cluster_membership, tree = raccoon.cluster(data, lab=labels, dim=2, pop_cut=20, max_depth=1,
                                     filter_feat='tSVD', optimizer='grid', metric_clu='euclidean', metric_map='cosine',
                                     dyn_mesh=True, max_mesh=3, min_mesh=3,
                                     out_path='./out_test_tsvd', save_map=True, debug=True, gpu=gpu)


def high_test(data, labels=None, gpu=False):
    """ Clustering test, cosine grid with >2 dimensions.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.    
    """

    cluster_membership, tree = raccoon.cluster(data, lab=labels, dim=3, pop_cut=20, max_depth=2,
                                     filter_feat='variance', optimizer='grid', metric_clu='cosine', metric_map='cosine',
                                     dyn_mesh=True, max_mesh=3, min_mesh=3,
                                     out_path='./out_test_high', save_map=True, debug=True, gpu=gpu)

def super_test(data, labels=None, gpu=False):
    """ Supervised clustering test, euclidean grid.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.
    """

    cluster_membership, tree = raccoon.cluster(data, lab=labels, supervised=True, dim=2, pop_cut=20, max_depth=1,
                                     filter_feat='variance', optimizer='grid', metric_clu='euclidean', metric_map='cosine',
                                     dyn_mesh=True, max_mesh=3, min_mesh=3,
                                     out_path='./out_test_super', save_map=True, debug=True, gpu=gpu)

def trans_test(data, labels=None, gpu=False):
    """ Clustering test, euclidean grid with transform-only data.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.    
    """

    cluster_membership, tree = raccoon.cluster(data, transform=data.sample(frac=.2).index, lab=labels, dim=2, pop_cut=20, max_depth=2,
                                     filter_feat='variance', optimizer='grid', metric_clu='euclidean', metric_map='cosine',
                                     dyn_mesh=True, max_mesh=3, min_mesh=3,
                                     out_path='./out_test_trans', save_map=True, debug=True, gpu=gpu)

def arand_test(data, truth, labels=None, gpu=False):
    """ Clustering test, euclidean grid, with rand index.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.
    """

    if gpu:
        from cuml.metrics.cluster.adjusted_rand_index import adjusted_rand_score as arand
    else:
        from sklearn.metrics import adjusted_rand_score as arand

    def arand_score(points, pred, metric=None):
        """ Example of external score to be fed to raccoon.
            Follows the format necessary to run with raccoon.

            Args:
                points (pandas dataframe, matrix): points coordinates, will be ignored.
                pred (pandas series): points labels obtained with raccoon.
                metric (string): distance metric, will be ignored.
         """
         
        return arand(truth.loc[pred.index], pred)

    cluster_membership, tree = raccoon.cluster(data, lab=labels, dim=2, pop_cut=20, max_depth=1, 
                                     filter_feat='variance', optimizer='grid', metric_clu='euclidean', metric_map='cosine',
                                     dyn_mesh=True, max_mesh=3, min_mesh=3, chk=False, score=arand_score,
                                     out_path='./out_test_arand', save_map=True, debug=True, gpu=gpu)

if __name__ == "__main__":

    pass
