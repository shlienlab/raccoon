"""
Set of standardized tests for the clustering function of RACCOON
F. Comitani     @2020
"""
import os
import sys
# tmp workarond
sys.path.append(r'/hpf/largeprojects/adam/projects/raccoon')

import pandas as pd

from raccoon.utils.classification import KNN

def knn_test(data, refpath, gpu=False):
    """ k-NN classification test, euclidean grid.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            refpath (string): path to reference files.
            gpu (bool): if True use gpu implementation.
    """
    
    rcknn = KNN(data.sample(frac=.5), data,
        pd.read_hdf(os.path.join(refpath,'raccoon_data/final_output.h5')),
        refpath=os.path.join(refpath,'raccoon_data'),
        outpath=refpath, 
        gpu=gpu)
    rcknn.assign_membership()

if __name__ == "__main__":

    pass
