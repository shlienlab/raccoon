"""
Utility functions for RACCOON (Recursive Algorithm for Coarse-to-fine Clustering OptimizatiON)
F. Comitani     @2018-2020
A. Maheshwari   @2019
"""

import numpy as np
import pandas as pd
import csv
import pickle
import os, sys, shutil, warnings

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import gaussian_kde
from scipy.stats import median_absolute_deviation as mad
from scipy.signal import argrelextrema
import logging

from raccoon.utils.plots import plotViolin


def _nearZeroVarDropAuto(data, thresh=0.99, type='variance'):

    """ Drop features with low variance/MAD based on a threshold after sorting them,
        converting to a cumulative function and keeping the 'thresh' % most variant features.

    Args:
        
        data (pandas dataframe): Input pandas dataframe (samples as row, features as columns).
        thresh (float): Percentage threshold for the cumulative variance/MAD.
        type (string): measure of variability, to be chosen between
            variance ('variance') or median absolute deviation ('MAD').
    """

    if type=='variance':
        vVal = data.var(axis=0).values
    elif type=='MAD':
        vVal = data.apply(mad,axis=0).values

    cs = pd.Series(vVal).sort_values(ascending=False).cumsum()
    remove = cs[cs > cs.values[-1]*thresh].index.values
    return data.drop(data.columns[remove], axis=1)


def _dropMinKDE(data, type='variance'):

    """ Use kernel density estimation to guess the optimal cutoff for low-variance removal. 

    Args:
        data (pandas dataframe): Input pandas dataframe (samples as row, features as columns).
        type (string): measure of variability, to be chosen between
            variance ('variance') or median absolute deviation ('MAD').

    """

    if type=='variance':
        vVal = data.var(axis=0).values
    elif type=='MAD':
        vVal = data.apply(mad,axis=0).values

    x = np.arange(np.min(vVal), np.max(vVal),
                  (np.max(vVal)-np.min(vVal))/100)
    kde = gaussian_kde(vVal, bw_method=None)
    y = kde.evaluate(x)

    imax = argrelextrema(y, np.greater)[0]
    imin = argrelextrema(y, np.less)[0]
    cutoff = None

    """ Take the last min before abs max. """

    absmax = np.argmax(y[imax])

    if absmax > 0:
        cutoff = x[np.max([xx for xx in imin if xx < imax[absmax]])]

    if cutoff != None:
        cs = pd.Series(vVal, index=data.columns)
        remove = cs[cs < cutoff].index.values
        data = data.drop(remove, axis=1)

    return data


def _calcRPD(mh, labs, plot=True, name='rpd', path=""):

    """ Calculate and plot the relative pairwise distance (RPD) distribution for each cluster.
        See XXX for the definition.
        WARNING: UNSTABLE, only works with cosine REMOVE IN FINAL VERSION

    Args:
        mh (pandas dataframe): Dataframe containing reduced dimensionality data.
        labs (pandas series): Clusters memebership for each sample.
        plot (boolean): True to generate plot, saves the RPD values only otherwise.
        name (string): Name of output violin plot .png file.

    Returns:
        vals (array of arrays of floats): Each internal array represents the RPD values of the corresponding cluster #

    """

    cosall=cosine_similarity(mh)
    if not isinstance(labs,pd.Series):
        labs=pd.Series(labs, index=mh.index)
    else:
        labs.index=mh.index

    csdf=pd.DataFrame(cosall, index=mh.index, columns=mh.index)
    csdf=csdf.apply(lambda x: 1-x)

    centroids=[]
    for i in set(labs.values):
        centroids.append(mh.iloc[np.where(labs==i)].mean())

    centroids=pd.DataFrame(centroids, index=set(labs.values), columns=mh.columns)
    coscen=cosine_similarity(centroids)
    coscen=pd.DataFrame(coscen, index=set(labs.values), columns=set(labs.values))
    coscen=coscen.apply(lambda x: 1-x)
    #np.fill_diagonal(coscen.values, 9999)

    vals=[]
    for i in set(labs.values):
        if i!=-1:

            matrix=csdf[labs[labs==i].index].loc[labs[labs==i].index].values

            siblings=set(labs.values)
            siblings.remove(i)
            siblings.discard(-1)

            """ take only the upper triangle (excluding the diagonal) of the matrix to avoid duplicates. """

            vals.append([x*1.0/coscen[i].loc[list(siblings)].min() for x in
                matrix[np.triu_indices(matrix.shape[0],k=1)]])


    with open(os.path.join(path,'raccoonData/rpd.pkl'), 'rb') as f:
        try:
            cur_maps = pickle.load(f)
        except EOFError:
            cur_maps = []

    for i in range(len(vals)):
        cur_maps.append([name + "_" + str(i),vals[i]])


    with open(os.path.join(path,'raccoonData/rpd.pkl'), 'wb') as f:  
        pickle.dump(cur_maps, f)

    if plot:
        plotViolin(vals, name, path)

    return vals

def setup(outpath=None, RPD=False):
   
    """ Set up folders that are written to during clustering, as well as a log file where all standard output is sent.
        If such folders are already present in the path, delete them.
 
    Args:
        outpath (string): Path where output files will be saved.
   
    """

    """ Build folders and delete old data if present. """

    try:
        os.makedirs(os.path.join(outpath, 'raccoonData'))
        os.makedirs(os.path.join(outpath, 'raccoonPlots'))
    except FileExistsError:
        warnings.warn('raccoonData/Plots already found in path!')
        answer=None
        while answer not in ['y','yes','n','no']:
            answer = input("Do you want to delete the old folders? [Y/N]  ").lower()
        if answer.startswith('y'):
            shutil.rmtree(os.path.join(outpath, 'raccoonData')) 
            os.makedirs(os.path.join(outpath, 'raccoonData'))
            shutil.rmtree(os.path.join(outpath, 'raccoonPlots')) 
            os.makedirs(os.path.join(outpath, 'raccoonPlots'))
        else:
            print('Please remove raccoonData/Plots manually or change output directory before proceeding!')
            sys.exit(1)


    """ Generate empty optimal paramaters table, to be written to at each iteration. """

    vals = ['name', 'n_samples', 'n_clusters', 'dim', 'silhouette_score',
        'n_neighbours', 'cluster_parm', 'genes_cutoff', 'metric_map', 'metric_clust', 'norm', 'reassigned']

    with open(os.path.join(outpath, 'raccoonData/paramdata.csv'), 'w') as file:
        writer = csv.writer(file)
        writer.writerow(vals)
        file.close()

    """ Generate empty calcRPD distributions pickle, to be written to at each iteration. """

    if RPD:
        with open(os.path.join(outpath, 'raccoonData/rpd.pkl'), 'wb') as file:
            empty = []
            pickle.dump(empty, file)
            file.close()

    """ Configure log. """

    logname='raccoon_'+str(os.getpid())+'.log'
    print('Log information will be saved to '+logname)

    logging.basicConfig(level=logging.INFO, filename=os.path.join(outpath, logname), filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.getLogger('matplotlib.font_manager').disabled = True
    
def sigmoid(x,a=0,b=1):

    """ Sigmoid function
 
    Args:
        x (float): position at which to evaluate the function
        a (float): center parameter
        b (float): slope parameter
   
    Returns:
        (float): sigmoid function evaluated at position x
    """

    return 1/(1+np.exp((x-a)*b))

