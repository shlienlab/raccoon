"""
Set of standardized tests for the clustering function of RACCOON
F. Comitani     @2020
"""

# tmp workarond
import sys
sys.path.append(r'/hpf/largeprojects/adam/projects/raccoon')

import raccoon.utils.trees as trees
import raccoon as rc


def grid_test(data, labels=None, gpu=False):
    """ Clustering test, euclidean grid, with checkpoints.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.
    """

    cluster_membership, tree = rc.run(data, lab=labels, dim=2, popcut=20, maxdepth=4,
                                     filterfeat='variance', optimizer='grid', metric_clu='euclidean', metric_map='cosine',
                                     dynmesh=True, maxmesh=3, minmesh=3, chk=True,
                                     outpath='./out_test_grid', savemap=True, debug=True, gpu=gpu)

def mahalanobis_test(data, labels=None, gpu=False):
    """ Clustering test, mahalanobis grid.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.
    """

    cluster_membership, tree = rc.run(data, lab=labels, dim=2, popcut=20, maxdepth=1,
                                     filterfeat='variance', optimizer='grid', metric_clu='mahalanobis', metric_map='cosine',
                                     dynmesh=True, maxmesh=3, minmesh=3,
                                     outpath='./out_test_mahalanobis', savemap=True, debug=True, gpu=gpu)

def snn_test(data, labels=None, gpu=False):
    """ Clustering test, euclidean grid, SNN DBSCAN.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.
    """

    cluster_membership, tree = rc.run(data, lab=labels, dim=2, popcut=20, maxdepth=1, clusterer='SNN',
                                     filterfeat='variance', optimizer='grid', metric_clu='euclidean', metric_map='cosine',
                                     dynmesh=True, maxmesh=3, minmesh=3,
                                     outpath='./out_test_snn', savemap=True, debug=True, gpu=gpu)

def louvain_test(data, labels=None, gpu=False):
    """ Clustering test, euclidean grid, SNN Louvain.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.
    """

    cluster_membership, tree = rc.run(data, lab=labels, dim=2, popcut=20, maxdepth=1, clusterer='louvain',
                                     filterfeat='variance', optimizer='grid', metric_clu='euclidean', metric_map='cosine',
                                     dynmesh=True, maxmesh=3, minmesh=3,
                                     outpath='./out_test_louvain', savemap=True, debug=True, gpu=gpu)

def load_test(data, load_path, labels=None, gpu=False):
    """ Clustering test, euclidean grid loading parameters data from file.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            load_path (strin): path to parameters data file to load.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.
    """

    cluster_membership, tree = rc.run(data, lab=labels, dim=2, popcut=20, maxdepth=1,
                                     filterfeat='variance', optimizer='grid', metric_clu='euclidean', metric_map='cosine',
                                     dynmesh=True, maxmesh=3, minmesh=3, fromfile=load_path,
                                     outpath='./out_test_load', savemap=True, debug=True, gpu=gpu)

    tree = trees.load_tree('./out_test_load/raccoon_data/tree.json')


def de_test(data, labels=None, gpu=False):
    """ Clustering test, euclidean Differential Evolution.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.    
    """

    cluster_membership, tree = rc.run(data, lab=labels, dim=2, popcut=10, maxdepth=2,
                                     filterfeat='variance', optimizer='de', metric_clu='euclidean', metric_map='cosine',
                                     dynmesh=True, maxmesh=4, minmesh=4,
                                     outpath='./out_test_de', savemap=True, debug=True, gpu=gpu)


def auto_test(data, labels=None, gpu=False):
    """ Clustering test, euclidean automatic selection.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.    
    """

    cluster_membership, tree = rc.run(data, lab=labels, dim=2, popcut=20, maxdepth=1,
                                     filterfeat='variance', optimizer='auto', metric_clu='euclidean', metric_map='cosine',
                                     dynmesh=True, maxmesh=3, minmesh=3,
                                     outpath='./out_test_auto', savemap=True, debug=True, gpu=gpu)


def tsvd_test(data, labels=None, gpu=False):
    """ Clustering test, euclidean grid with t-SVD.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.    
    """

    cluster_membership, tree = rc.run(data, lab=labels, dim=2, popcut=20, maxdepth=1,
                                     filterfeat='tSVD', optimizer='grid', metric_clu='euclidean', metric_map='cosine',
                                     dynmesh=True, maxmesh=3, minmesh=3,
                                     outpath='./out_test_tsvd', savemap=True, debug=True, gpu=gpu)


def high_test(data, labels=None, gpu=False):
    """ Clustering test, cosine grid with >2 dimensions.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.    
    """

    cluster_membership, tree = rc.run(data, lab=labels, dim=3, popcut=20, maxdepth=2,
                                     filterfeat='variance', optimizer='grid', metric_clu='cosine', metric_map='cosine',
                                     dynmesh=True, maxmesh=3, minmesh=3,
                                     outpath='./out_test_high', savemap=True, debug=True, gpu=gpu)

def super_test(data, labels=None, gpu=False):
    """ Supervised clustering test, euclidean grid.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.
    """

    cluster_membership, tree = rc.run(data, lab=labels, supervised=True, dim=2, popcut=20, maxdepth=1,
                                     filterfeat='variance', optimizer='grid', metric_clu='euclidean', metric_map='cosine',
                                     dynmesh=True, maxmesh=3, minmesh=3,
                                     outpath='./out_test_super', savemap=True, debug=True, gpu=gpu)

def trans_test(data, labels=None, gpu=False):
    """ Clustering test, euclidean grid with transform-only data.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            labels (pandas series, array): input test labels.
            gpu (bool): if True use gpu implementation.    
    """

    cluster_membership, tree = rc.run(data, transform=data.sample(frac=.2).index, lab=labels, dim=2, popcut=20, maxdepth=2,
                                     filterfeat='variance', optimizer='grid', metric_clu='euclidean', metric_map='cosine',
                                     dynmesh=True, maxmesh=3, minmesh=3,
                                     outpath='./out_test_trans', savemap=True, debug=True, gpu=gpu)

if __name__ == "__main__":

    pass
