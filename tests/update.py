"""
Set of standardized tests for the update function of RACCOON
F. Comitani     @2020-2021
"""
import os
import pandas as pd

import coon

def update_test(data, ori_data, reftab, refpath, gpu=False):
    """ Update clustering test, euclidean grid.

        Args:
            data (pandas dataframe, matrix): input test dataframe.
            ori_data (pandas dataframe, matrix): input test dataframe
                used for the original clustering.
            reftab (pandas dataframe, matrix): input test clustering
                assignment table.
            refpath (string): path to reference files.
            gpu (bool): if True use gpu implementation.
    """
    
    new_membership = coon.update(data, ori_data,
        pd.read_hdf(reftab), tolerance=1e-2,
        refpath=os.path.join(refpath,'raccoon_data'),
        dim=2, filterfeat='variance', optimizer='grid', 
        metric_clu='euclidean', metric_map='cosine',
        dynmesh=True, maxmesh=3, minmesh=3, chk=True,
        maxdepth=None, popcut=10, minclusize=5,
        outpath='./out_test_update', savemap=True, debug=True, 
        gpu=gpu)

if __name__ == "__main__":

    pass
