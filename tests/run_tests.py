"""
Set of standardized tests for RACCOON
F. Comitani     @2020
"""
from pathlib import Path
import shutil
from termcolor import colored

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
        shutil.rmtree(dirpath)


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

    x2, y2 = make_blobs(n_samples=100, centers=1, n_features=16,
                      random_state=32, cluster_std=10.0, center_box=(15,15))

    return pd.DataFrame(np.concatenate([x,x2])), pd.Series(np.concatenate([y,np.where(y2==0, 3, y2)]))

if __name__ == "__main__":


    print('Running Tests...')

    xx,yy = _createDataset()

    # """ Test Grid """

    # try:
    #     with hidePrints():
    #         grid_test(xx,labels=yy)
    #     print('Grid Test:\t\t'+colored('PASSED', 'green'))
    #     colored('PASSED', 'green')
    # except Exception as e:
    #     print('Grid Test:\t\t'+colored('FAILED', 'red'))
    #     print('An error occourred: ' + str(e))

    """ Test DE """

    try:
        with hidePrints():
            de_test(xx,labels=yy)
        print('DE Test:\t\t'+colored('PASSED', 'green'))
    except Exception as e:
        print('DE Test:\t\t'+colored('FAILED', 'red'))
        print('An error occourred: ' + str(e))

    # """ t-SVD Dimensions """

    # try:
    #     with hidePrints():
    #         tsvd_test(xx,labels=yy)
    #     print('t-SVD Test:\t\t'+colored('PASSED', 'green'))
    # except Exception as e:
    #     print('t-SVD Test:\t\t'+colored('FAILED', 'red'))
    #     print('An error occourred: ' + str(e))


    # """ Test High Dimensions """

    # try:
    #     with hidePrints():
    #         high_test(xx,labels=yy)
    #     print('High-dimensionality Test:\t\t'+colored('PASSED', 'green'))
    # except Exception as e:
    #     print('High-dimensionality Test:\t\t'+colored('FAILED', 'red'))
    #     print('An error occourred: ' + str(e))

    # """ Test Transform-only """

    # try:
    #     with hidePrints():
    #         trans_test(xx,labels=yy)
    #     print('Transform-only Test:\t\t'+colored('PASSED', 'green'))
    # except Exception as e:
    #     print('Transform-only Test:\t\t'+colored('FAILED', 'red'))
    #     print('An error occourred: ' + str(e))


    """ Test k-NN """

    try:
        with hidePrints():
            knn_test(xx, './outTest_de')
        print('k-NN Test:\t\t'+colored('PASSED', 'green'))
    except Exception as e:
        print('k-NN Test:\t\t'+colored('FAILED', 'red'))
        print('An error occourred: ' + str(e))        


    """ Clean up"""

    print('Cleaning up...')

    removeDir('./outTest_grid')
    removeDir('./outTest_de')
    removeDir('./outTest_tsvd')
    removeDir('./outTest_high')
    removeDir('./outTest_trans')

    print('All done!')