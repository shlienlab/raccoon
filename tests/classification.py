"""
Set of standardized tests for the clustering function of RACCOON
F. Comitani     @2020
"""
import os
import pandas as pd


#tmp workarond
import sys
sys.path.append(r'/Users/federico comitani/GitHub/raccoon')

from raccoon.utils.classification import knn

def knn_test(data,refpath):

    """ k-NN classification test, euclidean grid 

        Args:
            data (pandas dataframe, matrix): input test dataframe
            refpath (string): path to reference files
    """

    rcknn=knn(data.sample(frac=.5), data, pd.read_hdf(os.path.join(refpath,'finalOutput.h5')), refpath=os.path.join(refpath,'raccoonData'), outpath=refpath)
    rcknn.assignMembership()


if __name__ == "__main__":

    pass
