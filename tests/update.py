"""
Set of standardized tests for the update function of RACCOON
F. Comitani     @2020-2021
"""
import os
import pandas as pd

import raccoon

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
    
    new_membership = raccoon.update(data, ori_data,
        pd.read_hdf(reftab), tolerance=1e-2,
        refpath=os.path.join(refpath,'rc_data'),
        dim=2, filter_feat='variance', optimizer='grid', 
        metric_clu='euclidean', metric_map='cosine',
        dyn_mesh=True, max_mesh=3, min_mesh=3, chk=True,
        max_depth=None, pop_cut=15, min_csize=10,
        out_path='./out_test_update', save_map=True, debug=True, 
        gpu=gpu)

if __name__ == "__main__":

    pass
