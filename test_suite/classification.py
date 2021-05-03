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

def knn_test(data, reftab, refpath, gpu=False):
    """ k-NN classification test, euclidean grid.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            reftab (pandas dataframe, matrix): input test clustering
                assignment table.
            refpath (string): path to reference files.
            gpu (bool): if True use gpu implementation.
    """
   

    new_membership = rc.classify(data.sample(frac=.5), data,
        pd.read_hdf(reftab),
        #pd.read_hdf(os.path.join(refpath,'raccoon_data/clusters_final.h5')),
        refpath=os.path.join(refpath,'raccoon_data'),
        outpath=refpath, 
        gpu=gpu)

if __name__ == "__main__":

    pass
