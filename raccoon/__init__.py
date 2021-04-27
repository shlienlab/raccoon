
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
import warnings

import csv

import raccoon.utils.functions as functions
from raccoon.clustering import *
import raccoon.interface as interface

def run(data, **kwargs):
    """ Wrapper function to setup, create a RecursiveClustering object,
        run the recursion and logging.

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

    logging.info('Starting a new clustering run')
    
    """ Run recursive clustering algorithm. """

    obj = RecursiveClustering(data, **kwargs)
    obj.recurse()

    """ Save the assignment to disk and buil tree. """

    tree = None
    if obj.clus_opt is not None:
        obj.clus_opt.to_hdf(
            os.path.join(
                kwargs['outpath'], 'raccoon_data/clusters_final.h5'),
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


def resume(data, chkpath, **kwargs):

    """ Wrapper function to resume a RecursiveClustering run 
        from checkpoint files.

        Args:
            data (pandas dataframe): dataframe with sampels as rows and features as columns.
            chkpath (string): path to checkpoint files parent folder.
            **kwargs: keyword arguments for RecursiveClustering.

        Returns:
            clus_opt (pandas dataframe): one-hot-encoded clusters membership of data.
    """

    start_time = time.time()

    if 'outpath' in kwargs:
        warnings.warn('outpath is incompatible with resume, it will be ignored and chkpath will be used instead.')
    kwargs['outpath'] = chkpath
    if 'RPD' not in kwargs:
        kwargs['RPD'] = False
    if 'popcut' not in kwargs:
        kwargs['popcut'] = 50 #default value
    if 'maxdepth' not in kwargs:
        kwargs['maxdepth'] = None  #default value
    if 'depth' in kwargs:
        del kwargs['depth']

    """ Setup CPU/GPU interface. """

    if kwargs['gpu']:
        try:
            intf = interface.InterfaceGPU()
        except BaseException:
            warnings.warn("No RAPIDS found, running on CPU instead.")
            kwargs['gpu'] = False

    if not kwargs['gpu']:
        intf = interface.InterfaceCPU()

    """ Load parameters file. """

    try:
        oldparams = intf.df.read_csv(os.path.join(kwargs['outpath'],'raccoon_data/paramdata.csv'))
    except:
        sys.exit('ERROR: there was a problem loading the paramdata.csv, make sure the file path is correct.')    
        
    if oldparams.shape[0] == 0:
        sys.exit('ERROR: paramdata.csv is empty.')
   
    """ Load clustering files and join them. """    

    old_clus=[]
    for filename in os.listdir(os.path.join(kwargs['outpath'],'raccoon_data/chk')):
         old_clus.append(intf.df.read_hdf(os.path.join(kwargs['outpath'],'raccoon_data/chk',filename))\
            .reindex(data.index))
    
    if len(old_clus) == 0:
        sys.exit('ERROR: there was a problem loading the checkpoint files, make sure the file path is correct.\n'+\
                  'If no checkpoint files were generated repeat the run from scratch.')    

    elif len(old_clus) > 1: 
        # for now join not available in cudf concat 
        old_clus = intf.df.concat(
            old_clus, axis=1)
        old_clus = old_clus.fillna(0)
    
    else:
        old_clus = old_clus[0]

    old_clus=old_clus[functions.sort_len_num(intf.get_value(old_clus.columns))]
    new_clus=[old_clus]

    """ Setup logging."""

    functions.setup(kwargs['outpath'], kwargs['chk'], kwargs['RPD'], delete=False)

    logging.info('Resuming clustering run.')

    """ Fix discrepancies between parameters file and clusters. """

    to_drop = [x for x in oldparams.index if oldparams['name'].loc[x] != '0' and
               oldparams['name'].loc[x] not in old_clus.columns]

    if len(to_drop) > 0:
        logging.warning('Discrepancies between parameter file and clusters found in {:d} cases\n'.format(len(to_drop))+\
            'these will be automatically fixed.')
        
        oldparams.drop(to_drop,inplace=True)

    """ Identify clusters to resume. 
        Check population size, and depth. 
    """
    
    to_run = [x for x in old_clus.columns
                if x not in oldparams['name'].values and
                old_clus[x].sum()>kwargs['popcut'] and
                (kwargs['maxdepth'] is None or x.count('_') < kwargs['maxdepth'])]

    if len(to_run) == 0:
        logging.warning('No resumable cluster found. Your run might have completed succesfully.')
    
    else:

        """ Run recursive clustering algorithm. """

        for x in to_run:
            logging.info('Resuming cluster: '+x)
            
            cludata = data.loc[old_clus[x]==1]
            clulevel = x.count('_')

            obj = RecursiveClustering(cludata, depth=clulevel, name=x, **kwargs)
            obj.recurse()
            
            if obj.clus_opt is not None:
                new_clus.append(obj.clus_opt.astype(int))
    
            del obj

        if len(new_clus)>1:
            new_clus = intf.df.concat(new_clus,axis=1).fillna(0)
            new_clus = new_clus[functions.sort_len_num(intf.get_value(new_clus.columns))]
        else:
            new_clus = new_clus[0]

    """ Save the assignment to disk and buil tree. """

    tree = None
    if new_clus is not None:
        new_clus.to_hdf(
            os.path.join(
                kwargs['outpath'], 'raccoon_data/clusters_resumed_final.h5'),
            key='df')
        tree = trees.build_tree(
            new_clus, outpath=os.path.join(
                kwargs['outpath'], 'raccoon_data/tree_resumed_final.json'))

    """ Log the total runtime and memory usage. """

    logging.info('=========== Final Clustering Results ===========')
    if new_clus is not None:
        logging.info('A total of {:d} clusters were found'.format(
            len(new_clus.columns)))
    else:
        logging.info('No clusters found! Try changing the search parameters')
    logging.info(
        'Total time of the operation: {:.3f} seconds'.format(
            (time.time() - start_time)))
    logging.info(psutil.virtual_memory())

    return new_clus, tree


if __name__ == "__main__":

    pass
