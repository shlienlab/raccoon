"""
Set of standardized tests for the clustering function of RACCOON
F. Comitani     @2020
"""
import os
import sys
# tmp workarond
sys.path.append(r'/hpf/largeprojects/adam/projects/raccoon')

import pandas as pd
import raccoon as rc

def knn_test(data, ori_data, reftab, refpath, gpu=False):
    """ k-NN classification test, euclidean grid.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            ori_data (pandas dataframe, matrix): input test dataframe
                used for the original clustering.
            reftab (pandas dataframe, matrix): input test clustering
                assignment table.
            refpath (string): path to reference files.
            gpu (bool): if True use gpu implementation.
    """
   

    new_membership = rc.classify(data, ori_data,
        pd.read_hdf(reftab),
        refpath=os.path.join(refpath,'raccoon_data'),
        outpath='./out_test_knn',
        gpu=gpu)

if __name__ == "__main__":

    pass
