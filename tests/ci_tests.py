"""
Set of standardized tests for RACCOON
F. Comitani     @2020
"""

import sys
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from clustering import *
from classification import *
from update import *

def _create_dataset():
    """ Creates a dummy dataset for testing purposes.

        Returns:
            (matrix) cohordinates of dummy population.
            (array) cluster membership labels of dummy population.

    """


    x, y = make_blobs(n_samples=50, centers=4, n_features=16,
            random_state=32, cluster_std=1.0)

    x2, y2 = make_blobs(n_samples=25, centers=1, n_features=16,
            random_state=64, cluster_std=2.5, center_box=(-10, -10))

    x3, y3 = make_blobs(n_samples=25, centers=1, n_features=16,
            random_state=128, cluster_std=5)

    x4, y4 = make_blobs(n_samples=10, centers=1, n_features=16,
            random_state=0, cluster_std=.5, center_box=(5,5))
    x5, y5 = make_blobs(n_samples=10, centers=1, n_features=16,
            random_state=1, cluster_std=.25, center_box=(6, 6))
    x6, y6 = make_blobs(n_samples=10, centers=1, n_features=16,
            random_state=2, cluster_std=.25, center_box=(5.5, 5.5))

    
    return pd.DataFrame(np.concatenate([x, x2, x4, x5, x6]),
                index=[str(x)+'_o' for x in np.arange(x.shape[0]+x2.shape[0]+\
                    x4.shape[0]+x5.shape[0]+x6.shape[0])]),\
            pd.Series(np.concatenate([y, np.where(y2 == 0, 3, y2), np.where(y4 == 0, 4, y4),
                    np.where(y5 == 0, 5, y5), np.where(y6 == 0, 6, y6)]),
                index=[str(x)+'_o' for x in np.arange(x.shape[0]+x2.shape[0]+\
                    x4.shape[0]+x5.shape[0]+x6.shape[0])]),\
            pd.DataFrame(x3, 
                index=[str(x)+'_u' for x in np.arange(x3.shape[0])])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Raccoon Test Suite')
    parser.add_argument('-json', '-j', type=str, default='./testlist.json',
            help='tests selection list in json format (default: testlist.json)')
    args = parser.parse_args()    

    with open(args.json) as testlist:
        to_run = json.load(testlist)

    if to_run['gpu']:
        try:
            import cupy
            import cudf
            import cuml
            import cugraph
        except ImportError:
            print('An error occourred: ' + str(e))        
            print('GPU libraries (RAPIDS) not found!')        
            sys.exit(1)

    print('Running Tests...')

    xx, yy, xu = _create_dataset()

    """ Test Grid. """

    if to_run['grid']:
        grid_test(xx, labels = yy, gpu=to_run['gpu'])

    """ Test Mahalanobis. """

    if to_run['mahal']:
    	mahalanobis_test(xx, labels = yy, gpu=to_run['gpu'])

    """ Test SNN. """

    if to_run['snn']:
        snn_test(xx, labels = yy, gpu=to_run['gpu'])

    """ Test Louvain. """

    if to_run['louvain']:
    	louvain_test(xx, labels = yy, gpu=to_run['gpu'])

    """ Test Resume. """

    if to_run['resume']:
    	resume_test(xx, './out_test_grid', labels=yy, gpu=to_run['gpu'])

    """ Test DE. """

    if to_run['de']:
    	de_test(xx, labels = yy, gpu=to_run['gpu'])

    """ Test Auto. """

    if to_run['auto']:
    	auto_test(xx, labels = yy, gpu=to_run['gpu'])

    """ Test t-SVD. """

    if to_run['tsvd']:
    	tsvd_test(xx, labels = yy, gpu=to_run['gpu'])

    """ Test High Dimensions. """

    if to_run['high']:
    	high_test(xx, labels = yy, gpu=to_run['gpu'])

    """ Supervised Clustering. """

    if to_run['super']:
    	super_test(xx, labels = yy, gpu=to_run['gpu'])

    """ Test Transform-only. """

    if to_run['trans']:
    	trans_test(xx, labels = yy, gpu=to_run['gpu'])

    """ Test k-NN. """

    if to_run['knn']:
    	reftab='./out_test_grid/raccoon_data/clusters_final.h5'
    	knn_test(xu, xx, reftab, './out_test_grid', gpu=to_run['gpu'])
    
    """ Test k-NN. """

    if to_run['update']:
    	reftab='./out_test_grid/raccoon_data/clusters_final.h5'
    	update_test(xu, xx, reftab, './out_test_grid', gpu=to_run['gpu'])

