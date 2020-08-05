"""
Set of standardized tests for the clustering function of RACCOON
F. Comitani     @2020
"""

#tmp workarond
import sys
sys.path.append(r'/Users/federico comitani/GitHub/raccoon')

import raccoon as rc
import raccoon.utils.trees as trees

def grid_test(data,labels=None):

    """ Clustering test, euclidean grid 

        Args:
            data (pandas dataframe, matrix): input test dataframe
            lebels (pandas series, array): input test labels
    """

    clusterMembership, tree = rc.run(data, lab=labels, dim=2, popcut=20, maxdepth=2, 
                    filterfeat='variance', optimizer='grid', metricC='euclidean', metricM='cosine', 
                    dynmesh=True, maxmesh=3, minmesh=3, 
                    outpath='./outTest_grid', savemap=True, debug=True) 

def load_test(data,loadPath,labels=None):

    """ Clustering test, euclidean grid loading parameters data from file

        Args:
            data (pandas dataframe, matrix): input test dataframe
            loadPath (strin): path to parameters data file to load
            lebels (pandas series, array): input test labels
    """

    clusterMembership, tree = rc.run(data, lab=labels, dim=2, popcut=20, maxdepth=2, 
                    filterfeat='variance', optimizer='grid', metricC='euclidean', metricM='cosine', 
                    dynmesh=True, maxmesh=3, minmesh=3, fromfile=loadPath,
                    outpath='./outTest_load', savemap=True, debug=True) 
  
    tree = trees.loadTree('./outTest_load/tree.json')
    

def de_test(data,labels=None):

    """ Clustering test, euclidean Differential Evolution 

        Args:
            data (pandas dataframe, matrix): input test dataframe
            lebels (pandas series, array): input test labels
    """

    clusterMembership, tree = rc.run(data, lab=labels, dim=2, popcut=10, maxdepth=2, 
                    filterfeat='variance', optimizer='de', metricC='euclidean', metricM='cosine', 
                    dynmesh=True, maxmesh=4, minmesh=4, 
                    outpath='./outTest_de', savemap=True, debug=True) 


def auto_test(data,labels=None):

    """ Clustering test, euclidean automatic selection

        Args:
            data (pandas dataframe, matrix): input test dataframe
            lebels (pandas series, array): input test labels
    """

    clusterMembership, tree = rc.run(data, lab=labels, dim=2, popcut=20, maxdepth=1, 
                    filterfeat='variance', optimizer='auto', metricC='euclidean', metricM='cosine', 
                    dynmesh=True, maxmesh=3, minmesh=3, 
                    outpath='./outTest_auto', savemap=True, debug=True) 


def tsvd_test(data,labels=None):

    """ Clustering test, euclidean grid with t-SVD

        Args:
            data (pandas dataframe, matrix): input test dataframe
            lebels (pandas series, array): input test labels
    """

    clusterMembership, tree = rc.run(data, lab=labels, dim=2, popcut=20, maxdepth=1, 
                    filterfeat='tSVD', optimizer='grid', metricC='euclidean', metricM='cosine', 
                    dynmesh=True, maxmesh=3, minmesh=3, 
                    outpath='./outTest_tsvd', savemap=True, debug=True) 


def high_test(data,labels=None):

    """ Clustering test, cosine grid with >2 dimensions

        Args:
            data (pandas dataframe, matrix): input test dataframe
            lebels (pandas series, array): input test labels
    """

    clusterMembership, tree = rc.run(data, lab=labels, dim=3, popcut=20, maxdepth=1, 
                    filterfeat='variance', optimizer='grid', metricC='cosine', metricM='cosine', 
                    dynmesh=True, maxmesh=3, minmesh=3, 
                    outpath='./outTest_high', savemap=True, debug=True) 


def trans_test(data,labels=None):

    """ Clustering test, euclidean grid with transform-only data

        Args:
            data (pandas dataframe, matrix): input test dataframe
            lebels (pandas series, array): input test labels
    """

    clusterMembership, tree = rc.run(data, transform=data.sample(frac=.5).index, lab=labels, dim=2, popcut=20, maxdepth=2, 
                    filterfeat='variance', optimizer='grid', metricC='euclidean', metricM='cosine', 
                    dynmesh=True, maxmesh=3, minmesh=3, 
                    outpath='./outTest_trans', savemap=True, debug=True) 


if __name__ == "__main__":

    pass
