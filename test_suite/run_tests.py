"""
Set of standardized tests for RACCOON
F. Comitani     @2020
"""

import os,sys
import traceback
import shutil
from pathlib import Path
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from clustering import *
from classification import *

def colored_bak(string,color):
    return string
try:
    from termcolor import colored
except ImportError:
    colored=colored_bak

def remove_dir(path):
    """ Removes directory in path if it exists.

        Args:
            path (string): path to directory to remove
    """

    dirpath = Path(path)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath, ignore_errors=True)


class HidePrints:

    """ Temporarily hides standard outputs. """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def _create_dataset():
    """ Creates a dummy dataset for testing purposes.

        Returns:
            (matrix) cohordinates of dummy population.
            (array) cluster membership labels of dummy population.

    """

    x, y = make_blobs(n_samples=100, centers=3, n_features=16,
            random_state=32, cluster_std=1.0)

    x2, y2 = make_blobs(n_samples=50, centers=1, n_features=16,
            random_state=64, cluster_std=2.5, center_box=(10, 10))

    return pd.DataFrame(np.concatenate([x, x2])), pd.Series(
            np.concatenate([y, np.where(y2 == 0, 3, y2)]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Raccoon Test Suite')
    parser.add_argument('-json', '-j', type=str, default='./testlist.json',
            help='tests selection list in json format (default: testlist.json)')
    args = parser.parse_args()    


    #jsonpath = os.path.dirname(os.path.abspath(__file__))
    #with open(os.path.join(jsonpath, 'testlist.json')) as testlist:
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

    xx, yy = _create_dataset()

    if to_run['grid'] == False and (
            to_run['knn'] == True or to_run['resume'] == True):
        print('Warning: k-NN and Load test can\'t be run without Grid test')
        #to_run['knn'] = False
        #to_run['resume'] = False

    #if to_run['gpu'] == False and (to_run['knn_gpu'] == True):
    #    print('Warning: k-NN GPU can\'t be run without GPU test')
    #    to_run['knn_gpu'] = False

    """ Test Grid. """

    if to_run['grid']:
        try:
            #with HidePrints():
            grid_test(xx, labels = yy, gpu=to_run['gpu'])
            print('Grid Test:\t\t\t'+colored('PASSED', 'green'))
            colored('PASSED', 'green')
        except Exception as e:
            print('Grid Test:\t\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))
            traceback.print_exc()

    """ Test Mahalanobis. """

    if to_run['mahal']:
        try:
            #with HidePrints():
            mahalanobis_test(xx, labels = yy, gpu=to_run['gpu'])
            print('Mahalanobis Test:\t\t'+colored('PASSED', 'green'))
            colored('PASSED', 'green')
        except Exception as e:
            print('Mahalanobis Test:\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))
            traceback.print_exc()

    """ Test SNN. """

    if to_run['snn']:
        try:
            #with HidePrints():
            snn_test(xx, labels = yy, gpu=to_run['gpu'])
            print('SNN Test:\t\t\t'+colored('PASSED', 'green'))
            colored('PASSED', 'green')
        except Exception as e:
            print('SNN Test:\t\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))
            traceback.print_exc()

    """ Test Louvain. """

    if to_run['louvain']:
        try:
            #with HidePrints():
            louvain_test(xx, labels = yy, gpu=to_run['gpu'])
            print('Louvain Test:\t\t\t'+colored('PASSED', 'green'))
            colored('PASSED', 'green')
        except Exception as e:
            print('Louvain Test:\t\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))
            traceback.print_exc()

    """ Test Resume. """

    if to_run['resume']:
        try:
            #with HidePrints():
            resume_test(xx, './out_test_grid', labels=yy, gpu=to_run['gpu'])
            print('Load Test:\t\t\t'+colored('PASSED', 'green'))
            colored('PASSED', 'green')
        except Exception as e:
            print('Load Test:\t\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))
            traceback.print_exc()

    """ Test DE. """

    if to_run['de']:
        try:
            #with HidePrints():
            de_test(xx, labels = yy, gpu=to_run['gpu'])
            print('DE Test:\t\t\t'+colored('PASSED', 'green'))
        except Exception as e:
            print('DE Test:\t\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))
            traceback.print_exc()

    """ Test Auto. """

    if to_run['auto']:
        try:
            #with HidePrints():
            auto_test(xx, labels = yy, gpu=to_run['gpu'])
            print('Auto Test:\t\t\t'+colored('PASSED', 'green'))
        except Exception as e:
            print('Auto Test:\t\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))
            traceback.print_exc()

    """ Test t-SVD. """

    if to_run['tsvd']:
        try:
            #with HidePrints():
            tsvd_test(xx, labels = yy, gpu=to_run['gpu'])
            print('t-SVD Test:\t\t\t'+colored('PASSED', 'green'))
        except Exception as e:
            print('t-SVD Test:\t\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))
            traceback.print_exc()

    """ Test High Dimensions. """

    if to_run['high']:
        try:
            #with HidePrints():
            high_test(xx, labels = yy, gpu=to_run['gpu'])
            print('High-dimensionality Test:\t'+colored('PASSED', 'green'))
        except Exception as e:
            print('High-dimensionality Test:\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))
            traceback.print_exc()

    """ Supervised Clustering. """

    if to_run['super']:
        try:
            #with HidePrints():
            super_test(xx, labels = yy, gpu=to_run['gpu'])
            print('Supervised Clustering Test:\t'+colored('PASSED', 'green'))
        except Exception as e:
            print('Supervised Clustering Test:\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))
            traceback.print_exc()

    """ Test Transform-only. """

    if to_run['trans']:
        try:
            #with HidePrints():
            trans_test(xx, labels = yy, gpu=to_run['gpu'])
            print('Transform-only Test:\t\t'+colored('PASSED', 'green'))
        except Exception as e:
            print('Transform-only Test:\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))
            traceback.print_exc()

    """ Test k-NN. """

    if to_run['knn']:
        try:
            #with HidePrints():
            knn_test(xx, './out_test_grid', gpu=to_run['gpu'])
            print('k-NN Test:\t\t\t'+colored('PASSED', 'green'))
        except Exception as e:
            print('k-NN Test:\t\t\t'+colored('FAILED', 'red'))
            print('An error occourred: ' + str(e))        
            traceback.print_exc()

    """ Clean up. """

    if to_run['clean']:

        print('Cleaning up...')

        remove_dir('./out_test_grid')
        remove_dir('./out_test_mahalanobis')
        remove_dir('./out_test_snn')
        remove_dir('./out_test_louvain')
        #remove_dir('./out_test_resume')
        remove_dir('./out_test_de')
        remove_dir('./out_test_auto')
        remove_dir('./out_test_tsvd')
        remove_dir('./out_test_high')
        remove_dir('./out_test_super')
        remove_dir('./out_test_trans')

    print('All done!')
