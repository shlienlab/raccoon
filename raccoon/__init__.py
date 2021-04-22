
"""
RACCOON (Recursive Algorithm for Coarse-to-fine Clustering OptimizatiON)
F. Comitani     @2018-2021
A. Maheshwari   @2019
"""

import os

import logging
DEBUG_R = 15

import time
import psutil

import raccoon.utils.functions as functions
from raccoon.clustering import *

def run(data, **kwargs):
    """ Wrapper function to setup, create a RecursiveClustering object,
        run the recursion and logging in serial.

        Args:
            data (pandas dataframe): dataframe with sampels as rows and features as columns.
            **kwargs: keyword arguments for RecursiveClustering.

        Returns:
            clus_opt (pandas dataframe): one-hot-encoded clusters membership of data.
    """

    start_time = time.time()

    if 'outpath' not in kwargs or kwargs['outpath'] is None:
        kwargs['outpath'] = os.getcwd()
    if 'RPD' not in kwargs:
        kwargs['RPD'] = False

    """ Setup folders and files, remove old data if present. """

    functions.setup(kwargs['outpath'], kwargs['chk'], kwargs['RPD'])

    """ Run recursive clustering algorithm. """

    obj = RecursiveClustering(data, **kwargs)
    obj.recurse()

    """ Save the assignment to disk and buil tree. """

    tree = None
    if obj.clus_opt is not None:
        obj.clus_opt.to_hdf(
            os.path.join(
                kwargs['outpath'], 'raccoon_data/classes_final.h5'),
            key='df')
        tree = trees.build_tree(
            obj.clus_opt, outpath=os.path.join(
                kwargs['outpath'], 'raccoon_data/tree_final.json'))

    """ Log the total runtime and memory usage. """

    logging.info('=========== Final Clustering Results ===========')
    if obj.clus_opt is not None:
        logging.info('A total of {:d} clusters were found'.format(
            len(obj.clus_opt.columns)))
    else:
        logging.info('No clusters found! Try changing the search parameters')
    logging.info(
        'Total time of the operation: {:.3f} seconds'.format(
            (time.time() - start_time)))
    logging.info(psutil.virtual_memory())

    return obj.clus_opt, tree


if __name__ == "__main__":

    pass
