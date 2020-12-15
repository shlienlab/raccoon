"""
Set of standardized tests for RACCOON
F. Comitani     @2020
"""
from pathlib import Path
import os, shutil
from termcolor import colored
import json

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from clustering import *
from classification import *

def removeDir(path):

    """ Removes directory in path if it exists.

        Args:
            path (string): path to directory to remove
    """

    dirpath=Path(path)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath, ignore_errors=True)
    
class hidePrints:

    """ Temporarily hides standard outputs. """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def _createDataset():

    """ Creates a dummy dataset for testing purposes 

        Returns:
            (matrix) cohordinates of dummy population
            (array) cluster membership labels of dummy population
    
    """

    x, y = make_blobs(n_samples=100, centers=3, n_features=16,
                  random_state=32, cluster_std=1.0)

    x2, y2 = make_blobs(n_samples=50, centers=1, n_features=16,
                      random_state=32, cluster_std=10.0, center_box=(15,15))

    return pd.DataFrame(np.concatenate([x,x2])), pd.Series(np.concatenate([y,np.where(y2==0, 3, y2)]))

if __name__ == "__main__":

    jsonpath=os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(jsonpath,'testlist.json')) as testlist:
        to_run=json.load(testlist)

    print('Running Tests...')

    xx,yy = _createDataset()

    if to_run['grid'] == False and (to_run['knn'] == True or to_run['load'] == True):
        print('Warning: k-NN and Load test can\'t be run without Grid test')
        to_run['knn'] = False
        to_run['load'] = False
    
    if to_run['gpu'] == False and (to_run['knn_gpu'] == True):
        print('Warning: k-NN GPU can\'t be run without GPU test')
        to_run['knn_gpu'] = False

    # """ Test Grid """

    if to_run['grid']:
        try:
            with hidePrints():
                grid_test(xx, labels = yy)
            print('Grid Test:\t\t'+colored('PASSED', 'green'))
            colored('PASSED', 'green')
        except Exception as e:
            print('Grid Test:\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))

    # """ Test Load """

    if to_run['load']:
        try:
            with hidePrints():
                load_test(xx, './outTest_grid/raccoonData/paramdata.csv', labels=yy)
            print('Load Test:\t\t'+colored('PASSED', 'green'))
            colored('PASSED', 'green')
        except Exception as e:
            print('Load Test:\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))


    """ Test DE """

    if to_run['de']:
        try:
            with hidePrints():
                de_test(xx, labels = yy)
            print('DE Test:\t\t'+colored('PASSED', 'green'))
        except Exception as e:
            print('DE Test:\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))

    """ Test Auto """

    if to_run['auto']:
        try:
            with hidePrints():
                auto_test(xx, labels = yy)
            print('Auto Test:\t\t'+colored('PASSED', 'green'))
        except Exception as e:
            print('Auto Test:\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))


    """ t-SVD Dimensions """

    if to_run['tsvd']:
        try:
            with hidePrints():
                tsvd_test(xx, labels = yy)
            print('t-SVD Test:\t\t'+colored('PASSED', 'green'))
        except Exception as e:
            print('t-SVD Test:\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))


    """ Test High Dimensions """

    if to_run['high']:
        try:
            with hidePrints():
                high_test(xx, labels = yy)
            print('High-dimensionality Test:\t\t'+colored('PASSED', 'green'))
        except Exception as e:
            print('High-dimensionality Test:\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))

    """ Test Transform-only """

    if to_run['trans']:
        try:
            with hidePrints():
                trans_test(xx, labels = yy)
            print('Transform-only Test:\t\t'+colored('PASSED', 'green'))
        except Exception as e:
            print('Transform-only Test:\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))

    """ Test GPU """

    if to_run['gpu']:
        try:
            with hidePrints():
                #Test the import
                #to make sure rc doesn't just fall back to CPU in absence of RAPIDS
                from cuml import UMAP
                gpu_test(xx, labels = yy)
            print('GPU Test:\t\t'+colored('PASSED', 'green'))
        except Exception as e:
            print('GPU Test:\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))

    """ Test k-NN """

    if to_run['knn']:
        try:
            with hidePrints():
                knn_test(xx, './outTest_grid')
            print('k-NN Test:\t\t'+colored('PASSED', 'green'))
        except Exception as e:
            print('k-NN Test:\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))        

    """ Test k-NN with GPU """

    if to_run['knn_gpu']:
        try:
            with hidePrints():
                #Test the import
                #to make sure rc doesn't just fall back to CPU in absence of RAPIDS
                from cuml import UMAP
                knn_gpu_test(xx, './outTest_gpu')
            print('k-NN GPU Test:\t\t'+colored('PASSED', 'green'))
        except Exception as e:
            print('k-NN GPU Test:\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))        

    """ Clean up"""

    if to_run['clean']:
        
        print('Cleaning up...')
        
        removeDir('./outTest_grid')
        removeDir('./outTest_load')
        removeDir('./outTest_de')
        removeDir('./outTest_auto')
        removeDir('./outTest_tsvd')
        removeDir('./outTest_high')
        removeDir('./outTest_trans')
        removeDir('./outTest_gpu')

    print('All done!')
