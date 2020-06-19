"""
RACCOON (Recursive Algorithm for Coarse-to-fine Clustering OptimizatiON)
F. Comitani     @2018-2020
A. Maheshwari   @2019
"""

import os
import sys
import psutil

import pickle
import csv

import numpy as np
import pandas as pd

import umap
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD as tSVD
from scipy.sparse import csr_matrix
from hdbscan import HDBSCAN

import logging
import time
start_time = time.time()

#This not needed anymore
#from concurrent import futures

#TODO: see if .utils works

import raccoon.utils.plots as plotting
import raccoon.utils.functions as functions
import raccoon.utils.de as de
import raccoon.utils.classification 

""" Suppress UMAP and numpy warnings. """
import warnings
from numba.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", message="n_neighbors is larger than the dataset size; truncating to") 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

__version__ = "0.2.0"


class dataGlobal:

    """ Static container for the input data to be filled
        by the user at the first iteration. """

    #Hopefully this saves a bit of memory

    dataset=None


class recursiveClustering:

    """ To perform recursive clustering on a samples x features matrix. """

    def __init__(self, data, lab=None, transform=None, dim=2, epochs=5000, lr=0.05, neirange='logspace', neipoints=25, neifactor=1.0, 
        metricM='cosine', metricC='euclidean', popcut=50, filterfeat='variance', ffrange='logspace', 
        ffpoints=25, optimizer='grid', depop=10, deiter=10, score='silhouette', norm='l2', 
        clusterer='DBSCAN', cparmrange='guess', minclusize=10, outliers='ignore', 
        name='0', debug=False, maxdepth=None, savemap=False, RPD=False, outpath="", depth=-1, cores=1, _user=True):

        """ Initialize the the class.

        Args:
            data (matrix, pandas dataframe or pandas index): If first call (_user==True), input data in pandas dataframe-compatible format (samples as row, features as columns),
                otherwise index of samples to carry downstream during the recursion calls.
            lab (list, array or pandas series): List of labels corresponding to each sample (for plotting only).
            transform (list of Pandas DataFrame indices): List of indices of the samples in the initial matrix that should be transformed-only 
                and not used for training the dimensionality reduction map.
            dim (integer): Number of dimensions of the target projection (default 2).
            epochs (integer): Number of UMAP epochs.
            lr (float): UMAP learning rate.
            neirange (array, list of integers or string): List of nearest neighbors values to be used in the search;
                if 'logspace' take an adaptive range based on the dataset size at each iteration with logarithmic spacing (reccomended),
                if 'sqrt' always take the sqare root of the number of samples.
                if 'quartsqrt' always take the sqare root of half the number of samples (ideal for extremely large datasets).
            neipoints (int or list of int): Number of grid points for the neighbors search,  
                if list, each value will be subsequently used at the next iteration until all values are exhausted, 
                (works only with neirange='logspace' default 25).
            neifactor (float): Scaling factor for 'logspace' and 'sqrt' selections in neirange
            metricM (string): Metric to be used in UMAP distance calculations (default cosine).
            metricC (string): Metric to be used in clusters identification and Clustering score calculations (default euclidean)
                Warning: cosine does not work with HDBSCAN, normalize to 'l2' and use 'euclidean' instead.
            popcut (integer): Minimum number of samples for a cluster to be considered valid (default 50).
            filterfeat (string): Set the method to filter features in preprocessing;
                if 'variance' remove low variance genes
                if 'MAD' remove low median absolute deviation genes
                if 'tSVD' use truncated single value decomposition (LSA)
            ffrange (array, list or string): if filterfeat=='variance', percentage values for the low-variance removal cufoff search;
                if 'logspace' (default) take a range between .3 and .9 with logarithmic spacing 
                    (reccomended, will take the extremes if optimizer=='de');
                if 'kde' kernel density estimation will be used to find a single optimal low-variance cutoff (not compatible with optimizer=='de')
                if filterfeat=='tSVD', values for the number of output compontents search;
                if 'logspace' (default) take a range between number of features times .3 and .9 with logarithmic spacing 
                    (reccomended, will take the extremes if optimizer=='de')
            ffpoins (int or list of int): Number of grid points for the feature removal cutoff search  
                if list, each value will be subsequently used at the next iteration until all values are exhausted, 
                (works only with ffrange='logspace', default 25).
            optimizer (string): Choice of parameters optimizer, can be either 'grid' for grid search or 'de' for differential evolution (default is 'grid').
            depop (int or list of int): Size of the candidate solutions population in differential evolution  
                if list, each value will be subsequently used at the next iteration until all values are exhausted
                (works only with optimizer='de', default 10).
            deiter (int or list of int): Maximum number of iterations of differential evolution  
                if list, each value will be subsequently used at the next iteration until all values are exhausted
                (works only with optimizer='de', default 10).
            score (string): objective function of the optimization (default 'silhouette').    
            norm (string): normalization factor before dimensionality reduction (default 'l2'), not needed if metricM is cosine
                if None, don't normalize.
            clusterer (string): selects which algorithm to use for clusters identification. Choose between 'DBSCAN' (default) or HDBSCAN
            cparmrange (array, list) or string: clusters identification parameter range to be explored (default 'guess'). 
                When 'DBSCAN' this corresponds to epsilon (if 'guess' attempts to identify it by the elbow method);
                When 'HDBSCAN' this corresponds to the minimum number of samples required by the clusters (if 'guess' adapts it on the 
                    dataset population).
            minclusize (int): minimum number of samples in a cluster used in DBSCAN and HDBSCAN (default is 10).  
            outliers (string): selects how to deal with outlier points in the clusters assignment
                if 'ignore' discard them
                if 'reassign' try to assign them to other clusters with knn if more than 10% of the total population was flagged 
            name (string): Name of current clustering level (should be left as default, '0', unless continuing from a previous run).
            debug (boolean): Specifies whether algorithm is run in debug mode (default is False).
            maxdepth (int): Specify the maximum number of recursion iterations, if None (default), keep going while possible. 
                0 stops the algorithm immediately, 1 stops it after the first level.
            savemap (boolean): If active, saves the trained maps to disk (default is False). Needed to run the k-NN classifier.
            RPD (boolean): Specifies whether to save RPD distributions for each cluster (default is False). WARNING: this option is unstable
                and not reccomended.
            outpath (string): Path to the location where outputs will be saved (default, save to the current folder).
            depth (integer): Current depth of recursion (should be left as default, -1, unless continuing from a previous run).
            _user (bool): Active switch to separate initial user input versus recursion calls, do not change.
            #### TO REMOVE cores (int): Specifies how many cores to use in the parallel version (this value has no affect when run in serial).
        

        """

        if _user:

            if not isinstance(data, pd.DataFrame):
                try:
                    data=pd.DataFrame(data)
                except:
                    print('Unexpected error: ', sys.exc_info()[0])
                    print('Input data should be in a format that can be translated to pandas dataframe!')
                    raise

            dataGlobal.dataset=data
            data=data.index

        if lab is not None and not isinstance(lab, pd.Series):
            try:
                lab=pd.Series(lab)
            except:
                print('Unexpected error: ', sys.exc_info()[0])
                print('Labels data should be in a format that can be translated to pandas series!')
                raise
            try:
                lab.index=data.index
            except:
                print('Unexpected error: ', sys.exc_info()[0])
                print('Shape of labels data should be consistent with the shape of data!')
                raise


        self.dataIx = data
        self.lab = lab
        self.transform = transform
        
        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.neirange = neirange
        self.neipoints = neipoints
        self.neifactor = neifactor
        self.metricM = metricM
        self.metricC = metricC
        self.popcut = popcut 
        self.filterfeat = filterfeat
        self.ffrange = ffrange 
        self.ffpoints= ffpoints
        self.debug = debug 
        self.savemap = savemap
        self.maxdepth = maxdepth
        self.RPD = RPD
        self.optimizer = optimizer
        self.depop = depop
        self.deiter = deiter
        self.outpath= outpath
        self.clusOpt= None
        self._name = name
        self._depth = depth
        self.score = score
        self.norm = norm

        self.clusterer= clusterer
        self.cparmrange = cparmrange
        self.minclusize = minclusize
        self.outliers = outliers

        #to remove
        self._cores = cores

        """ Checks on optimizer choice """        

        if self.optimizer not in ['grid','de']:
            sys.exit('ERROR: Optimizer must be either \'grid\' for Grid Search or \'de\' for Differential Evolution.')

        if self.optimizer=='de' and self.ffrange=='kde':
            sys.exit('ERROR: KDE estimation of the low variance/MAD removal cutoff is not compatible with Differential Evolution.')

        if self.filterfeat not in ['variance','MAD','tSVD']:
            sys.exit('ERROR: Features filter must be either \'variance\' for low-variance removal, \'MAD\' for low-MAD removal or \'tSVD\' for truncated SVD.')

        """ Evaluate parameters granularity options """

        try:
            if isinstance(self.neipoints, list):
                self.neipoints=[int(x) for x in self.neipoints]
            else:
                self.neipoints=[int(self.neipoints)]
        except:
            sys.exit('ERROR: neipoints must be an integer or a list of integers')
            raise

        if self.optimizer=='de':
            try:
                if isinstance(self.depop, list):
                    self.depops=[int(x) for x in self.depop]
                else:
                    self.depop=[int(self.depop)]
            except:
                sys.exit('ERROR: depop must be an integer or a list of integers')
                raise
            try:
                if isinstance(self.deiter, list):
                    self.deiter=[int(x) for x in self.deiter]
                else:
                    self.deiter=[int(self.deiter)]
            except:
                sys.exit('ERROR: deiter must be an integer or a list of integers')
                raise
            if self.ffrange == 'logspace':
                if self.filterfeat in ['variance','MAD']:
                    self.ffrange = [0.3,0.9]
                if self.filterfeat=='tSVD':
                    self.ffrange = [int(np.min([50,dataGlobal.dataset.loc[self.dataIx].shape[1]*0.3])),int(dataGlobal.dataset.loc[self.dataIx].shape[1]*0.9)]

        if self.optimizer=='grid':
            try:
                if isinstance(self.ffpoints, list):
                    self.ffpoints=[int(x) for x in self.ffpoints]
                else:
                    self.ffpoints=[int(self.ffpoints)]
            except:
                sys.exit('ERROR: ffpoints must be an integer or a list of integers')
                raise
            if self.ffrange == 'logspace':
                if self.filterfeat in ['variance','MAD']:
                    self.ffrange = sorted([float(x) for x in np.logspace(np.log10(0.3), np.log10(0.9), num=self.ffpoints[0])])
                if self.filterfeat=='tSVD':
                    self.ffrange = sorted([int(x) for x in np.logspace(np.log10(np.min([50,dataGlobal.dataset.loc[self.dataIx].shape[1]*0.3])), np.log10(dataGlobal.dataset.loc[self.dataIx].shape[1]*0.9), num=self.ffpoints[0])])
                    

        """ Setup logging """ 

        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            self._umapRs=32
        else:
            logging.getLogger().setLevel(logging.INFO)
            self._umapRs=None


    def _featuresRemoval(self, cutoff):


        """ Either remove features with low variance/MAD from dataset according to a specified threshold (cutoff) 
            or apply truncated SVD to reduce features to a certain number (cutoff).

        Args:
             cutoff (string or float): if filterfeat=='variance' or 'MAD', percentage value for the low-variance/MAD removal cufoff,
                if 'kde' kernel density estimation will be used to find a single optimal low-variance/MAD cutoff;
                if filterfeat=='tSVD', dimensionality of the output data. 

        Returns:
            (pandas dataframe): reduced-dimensionality input data.
            (tsvd object): trained tsvd instance, None if 'variance'/'MAD'.

        """

        if self.filterfeat=='tSVD':

            logging.debug("Applying Truncated SVD with #{:d} features".format(int(cutoff)))

            #sparseMat=csr_matrix(dataGlobal.dataset.loc[self.dataIx].values)
            decomposer=tSVD(n_components=int(cutoff))

            """ Add conditional to apply the cut only on those samples used for training the map. """

            #Ugly but hopefully more memory efficient
            if self.transform is not None:
                decomposer.fit(csr_matrix(dataGlobal.dataset.loc[self.dataIx][~dataGlobal.dataset.loc[self.dataIx].index.isin(self.transform)].values))
                return pd.DataFrame(decomposer.transform(csr_matrix(dataGlobal.dataset.loc[self.dataIx].values)), index=dataGlobal.dataset.loc[self.dataIx].index), decomposer
            else:
                return pd.DataFrame(decomposer.fit_transform(csr_matrix(dataGlobal.dataset.loc[self.dataIx].values)), index=dataGlobal.dataset.loc[self.dataIx].index), decomposer
            #return 
        
        elif self.filterfeat in ['variance','MAD']:

            """ Add conditional to apply the cut only on those samples used for training the map. """
            if self.transform is not None:
                newData = dataGlobal.dataset.loc[self.dataIx][~dataGlobal.dataset.loc[self.dataIx].index.isin(self.transform)]
            else: 
                newData = dataGlobal.dataset.loc[self.dataIx] 

            if cutoff == 'kde':
                newData=functions._dropMinKDE(newData, type=self.filterfeat)
            else:
                newData=functions._nearZeroVarDropAuto(newData,cutoff, type=self.filterfeat)

            
            logging.debug("Dropped Features #: " +
                      '{:1.0f}'.format(dataGlobal.dataset.loc[self.dataIx].shape[1]-newData.shape[1]))

            # Extra passage needed in case the transform data cut was applied        
            return dataGlobal.dataset.loc[self.dataIx][newData.columns], None

    def _levelCheck(self):

        """ Stop the recursion if a given maxdepth parameter has been reached. """

        if self.maxdepth is not None:
            self.maxdepth -= 1
            if (self.maxdepth == -1):
                self.clusOpt = None
                return True
            else:
                return False


    def _plot(self, nNei, proj, cutOpt, keepfeat, decomposer, clusOpt, parmvals):

        """ Produce a number of plots to visualize the clustering outcome at each stage of the recursion.

        Args:
            n_nei (integer): Optimal number of nearest neighbors (used in UMAP) that was found through grid search.
            proj (pandas dataframe of floats): Optimal reduced dimensionality data matrix. 
            cutOpt (int or float): Optimal features removal cutoff.
            keepfeat (pandas index): Set of genes kept after low /MAD removal, nan if tSVD.
            decomposer (tsvd object): trained tsvd instance.
            clusOpt (pandas series): Cluster membership series.
            parmvals (list of float): List of all scores evaluated. 

        """

        """ Plot the score values obtained during the search """
        try:
            plotting._plotScore(parmvals, self.cparmrange, 'scores_'+self._name, self.outpath)
        except:
            logging.warning('Failed to plot clustering scores')


        """ Plot the Relative Pairwise Distance (RPD) distributions. """

        if self.RPD:
            #WARNING: unstable
            try:
                functions._calcRPD(proj, clusOpt, True, self._name, self.outpath) 
            except:
                logging.warning('RPD failed at step: '+self._name)

        if self.filterfeat in ['variance','MAD'] and isinstance(keepfeat,pd.Index):
            selcut = dataGlobal.dataset.loc[self.dataIx][keepfeat]

            """ Plots distribution of variance/MAD and low-variance/MAD genes cutoff. """

            plotting._plotCut(dataGlobal.dataset.loc[self.dataIx], selcut, 'cut_'+self._name, self.outpath)

        elif self.filterfeat=='tSVD' and decomposer is not None:
            selcut=pd.DataFrame(decomposer.transform(csr_matrix(dataGlobal.dataset.loc[self.dataIx].values)), index=dataGlobal.dataset.loc[self.dataIx].index)
            #selcut = self._featuresRemoval(int(cutOpt))
        else:
            selcut = dataGlobal.dataset.loc[self.dataIx]

        if proj.shape[1]!=2:
            mapping = umap.UMAP(metric=self.metricM, n_components=2, min_dist=0.05, spread=1, n_neighbors=nNei,
                                n_epochs=self.epochs, learning_rate=self.lr,
                                verbose=False)
            
            if self.transform is not None:
                scft=selcut[~selcut.index.isin(self.transform)]
                sct=selcut[selcut.index.isin(self.transform)]
                mapping.fit(scft)
                proj = pd.concat([pd.DataFrame(mapping.transform(scft), index=scft.index), 
                                  pd.DataFrame(mapping.transform(sct),index=sct.index)], axis=0)
                proj = proj.loc[selcut.index]
            else: 
                proj = pd.DataFrame(mapping.fit_transform(selcut), index=selcut.index) 


            if (self.savemap == True):
                with open(os.path.join(self.outpath,'raccoonData/'+self._name+'_2d.pkl'), 'wb') as file:
                    #keepfeat and decompt already in the not 2d map
                    pickle.dump(mapping, file)
                    file.close()
                proj.to_hdf(os.path.join(self.outpath,'raccoonData/'+self._name+'_2d.h5'), key='proj')

        """ Plot 2-dimensional umap of the optimal clusters. """

        plotting.plotMap(proj, clusOpt, 'proj_clusters_'+self._name, self.outpath)

        """ Plot the same 2-dimensional umap with labels if provided. """

        if self.lab is not None:
            plotting.plotMap(proj, self.lab, 'proj_labels_'+self._name, self.outpath)

        """ Plot the same 2-dimensional umap with transform only data if provided. """

        if self.transform is not None:
            transflab=pd.Series('fit-transform',index=proj.index)
            transflab.loc[self.transform]='transform'
            #transflab.loc[set(self.transform).intersection(proj.index.values)]='transform'
            plotting.plotMap(proj, transflab, 'proj_trans_'+self._name, self.outpath)
        

    def _binarize(self, labsOpt, minpop=10):

        """ Construct and return a one-hot-encoded clusters membership dataframe.

        Args:
            labsOpt (pandas series): cluster membership series or list.

        Returns:
            tmplab (pandas dataframe): one-hot-encoded cluster membership dataframe.

        """

        label_binarizer = LabelBinarizer() 

        #NOTE: Extra column needed in case labsOpt is already binary (will be removed by drop if empty)
        label_binarizer.fit(range(len(list(set(labsOpt)))+1))
        tmplab = pd.DataFrame(label_binarizer.transform(labsOpt))

        """ Discard clusters that have less than minpop of population. """

        tmplab.drop(tmplab.columns.values[tmplab.sum() < minpop], axis=1, inplace=True) 
        tmplab = tmplab.set_index(dataGlobal.dataset.loc[self.dataIx].index.values)
        tmplab.columns = [self._name + "_" + str(x) for x in range(len(tmplab.columns.values))]
       
        return tmplab


    def _elbow(self, pj):

        """ Estimates the point of flex of a pairwise distances plot.

        Args:
            pj (pandas dataframe): Projection of saxmples in the low-dimensionality space obtained with UMAP.

        Returns:
            (float): elbow value.
        """

        neigh=NN(n_neighbors=2, metric=self.metricC, n_jobs=-1).fit(pj)
        neigh=np.sort(neigh.kneighbors(pj, return_distance=True)[0][:,1])
        neigh=pd.DataFrame(neigh,columns=['elbow'])
        neigh['delta']=neigh['elbow'].diff().shift(periods=-1)+-1*neigh['elbow'].diff()
        return neigh['elbow'].iloc[neigh['delta'].idxmin()]

    def _guessParm(self, pj):

        """ Estimate a range for the clustering identification parameter.

        Args:
            pj (pandas dataframe): Projection of saxmples in the low-dimensionality space obtained with UMAP.

        Returns:
            (numpy range): estimated range.
        """

        """ Use pariwise knn distances elbow method for DBSCAN;
            Take the square root of the total population for HDBSCAN"""

        if self.clusterer=='DBSCAN':
            ref=self._elbow(pj)
            logging.info('Epsilon range guess: [{:.5f},{:.5f}]'.format(ref/50,ref*1.5))

            return np.arange(ref/50,ref*1.5,(ref*1.5-ref/50)/100.0)

        elif self.clusterer=='HDBSCAN':
            minbound=np.max([self.minclusize,int(np.sqrt(pj.shape[0]/25))])
            maxbound=np.min([250,int(np.sqrt(pj.shape[0]*2.5))])
            if minbound==maxbound:
                maxbound=minbound+2

            step=int((maxbound-minbound)/50)
            if step<1:
                step=1
            logging.info('Minimum samples range guess: [{:d},{:d}] with a {:d} point(s) step'.format(minbound, maxbound, step)) 
            
            return np.arange(minbound, maxbound, step)

        else: 
            sys.exit('ERROR: clustering algorithm not recognized')


    def calcScore(self, points, labels):
       
        """ Select and calculate scoring function for optimization.

        Args:
            points (dataframe or matrix): points coordinates
            labels (series or matrix): clusters assignment

        Returns:
            (float): clustering score

        """

        if self.score=='silhouette':
            return metrics.silhouette_score(points, labels, metric=self.metricC)
        else: 
            sys.exit('ERROR: score not recognized')


    def _findClusters(self, pj, cparm, cse=None, algorithm=None):

        """ Runs the selected density-based clusters identification algorithm.

        Args:
            pj (dataframe or matrics): points coordinates
            cparm (float): clustering parameter
            cse (int): value of clustering_selection_epsilon for HDBSCAN
            algorithm (string): value of algorithm for HDBSCAN

        Returns:
            (list of int): list of assigned clusters """

        if self.clusterer=='DBSCAN':
            return DBSCAN(eps=cparm, min_samples=self.minclusize, metric=self.metricC, n_jobs=-1, leaf_size=15).fit_predict(pj)
        elif self.clusterer=='HDBSCAN':
            clusterer=HDBSCAN(algorithm=algorithm, alpha=1.0, approx_min_span_tree=True,
                    gen_min_span_tree=False, leaf_size=15, allow_single_cluster=False,
                    metric=self.metricC, min_cluster_size=self.minclusize, min_samples=int(cparm), 
                    cluster_selection_epsilon=cse, p=None).fit(pj)
            return clusterer.labels_
        else:
            sys.exit('ERROR: clustering algorithm not recognized')


    def _runSingleInstance(self, cutoff, nn):

        """ Run a single instance of clusters search for a given features cutoff and UMAP nearest neighbors number.

        Args:
            cutoff (float): features cutoff.
            nn (int): UMAP nearest neighbors value.

        Returns:
            silOpt (float): Silhoutte score corresponding to the best set of parameters.
            labs (pandas series): Series with the cluster membership identified for each sample.
            cparmOpt (float): Optimal clustering parameter value found.
            pj (pandas dataframe): Low dimensionality data projection from UMAP.
            keepfeat (pandas index): Set of genes kept after low /MAD removal, nan if 'tSVD'.
            decomposer (tsvd object): trained tsvd instance, None if 'variance'/'MAD'.
            parmvals (list of float): List of all scores evaluated. 

        """

        silOpt=-0.0001
        labsOpt = [0]*dataGlobal.dataset.loc[self.dataIx].shape[0]
        cparmOpt = np.nan
        keepfeat = np.nan
        parmvals=[]

        init='spectral'
        if dataGlobal.dataset.loc[self.dataIx].shape[0]<=self.dim:
            init='random'

        """ Remove columns with low information from data matrix. """

        logging.debug('Features cutoff: {:.3f}'.format(cutoff))

        dataCut, decomposer = self._featuresRemoval(cutoff)

        if self.filterfeat in ['variance','MAD']:
            keepfeat= dataCut.columns


        if self.norm is not None:

            """ Normalize data. """

            logging.debug('Normalize with '+self.norm)

            dataCut=pd.DataFrame(normalize(dataCut, norm=self.norm), index=dataCut.index, columns=dataCut.columns)

        """ Project data with UMAP. """        

        logging.debug('Number of nearest neighbors: {:d}'.format(nn))

        mapping = umap.UMAP(metric=self.metricM, n_components=self.dim, min_dist=0.0, spread=1, n_neighbors=nn,
                n_epochs=self.epochs, learning_rate=self.lr, verbose=False, random_state=self._umapRs, init=init)
        
        if self.transform is not None:
            pj = pd.DataFrame(mapping.fit_transform(dataCut[~dataCut.index.isin(self.transform)]), 
                index=dataCut[~dataCut.index.isin(self.transform)].index)
        else: 
            pj = pd.DataFrame(mapping.fit_transform(dataCut), index=dataCut.index) 

        if not pj.isnull().values.any():

            """ Set cluster_selection_epsilon for HDBSCAN """

            cse=None        
            hdbalgo='best'    
            if self.clusterer=='HDBSCAN':
                cse=float(self._elbow(pj))
                if self.metricC=='cosine':
                    hdbalgo='generic'
                    pj=pj.astype(np.float64)
            parmvals.append([])

            """ Set clustering parameter range at the first iteration. """

            #if self.cparmrange=='guess':

                #self.cparmrange=self._guessEps(pj.sample(np.min([500,pj.shape[0]])))
            #    self.cparmrange=self._guessParm(pj)

            #TODO: check if better to update at every iteration
            cparmrange=self.cparmrange
            if cparmrange=='guess':
                cparmrange=self._guessParm(pj)

            #Note: Calculating (H)DBSCAN on a grid of parameters is cheap even with Differential Evolution.

            for cparm in cparmrange: #self.cparmrange

                logging.debug('Clustering parameter: {:.5f}'.format(cparm))

                labs = self._findClusters(pj, cparm, cse, hdbalgo)

                #not 100% sure about this, keep until weights on noise will be added
                compset=set(labs)
                compset.discard(-1)

                if len(compset) > 1 :  
                    sil = self.calcScore(pj, labs)
                else:
                    sil = 0
                
                parmvals[-1].append(sil)

                if sil > silOpt:
                    cparmOpt = cparm
                    silOpt= sil
                    labsOpt = labs

        return silOpt, pd.Series(labsOpt, index=pj.index), cparmOpt, pj, mapping, keepfeat, decomposer, parmvals


    def _objectiveFunction(self, params):

        """ Objective function for Differential Evolution.

        Args:
            params (list): A list containing a single feature cutoff and a UMAP nearest neighbors parameter.
        Returns:
            (float): Loss value for the given set of parameters.
            labs (pandas series): Series with the cluster membership identified for each sample.
            cparmOpt (float): Optimal clustering parameter value found.
            pj (pandas dataframe): Low dimensionality data projection from UMAP.
            keepfeat (pandas index): Set of genes kept after low variance/MAD removal, nan if tSVD.
            decomposer (tsvd object): trained tsvd instance, None if 'variance'/'MAD'.
            parmvals (list of float): List of all scores evaluated. 

        """

        silOpt, labs, cparmOpt, pj, mapping, keepfeat, decomposer, parmvals = self._runSingleInstance(params[0],int(params[1]))
        
        return 1-silOpt, labs, cparmOpt, pj, mapping, keepfeat, decomposer, parmvals


    def _runGridInstances(self, nnrange):

        """ Run Grid Search to find the optimal set of parameters by maximizing the Clustering Score.

        Args:
            nnrange (numpy range): UMAP nearest neighbors range.

        Returns:
            silOpt (float): Silhoutte score corresponding to the best set of parameters.
            labsOpt (pandas series): Series with the cluster membership identified for each sample.
            cparmOpt (float): Optimal clustering parameter value found.
            numClusOpt (int):  Total number of clusters determined by the search.
            neiOpt (int): Optimal number of nearest neighbors used with UMAP.
            pjOpt (pandas dataframe): Low dimensionality data projection from UMAP.
            cutOpt (float): Optimal cutoff value used for the feature removal step
            keepfeat (pandas index): Set of genes kept after low variance/MAD removal, nan if tSVD.
            decompOpt (tsvd object): trained tsvd instance, None if 'variance'/'MAD'.
            parmvals (list of float): List of all scores evaluated. 

        """

        silOpt=-0.0001
        keepfeat = np.nan
        decompOpt = None
        labsOpt = [0]*dataGlobal.dataset.loc[self.dataIx].shape[0]
        cparmOpt = np.nan
        neiOpt = dataGlobal.dataset.loc[self.dataIx].shape[0]
        if self.filterfeat in ['variance','MAD']:
            cutOpt = 1.0
        elif self.filterfeat=='tSVD':
            cutOpt = dataGlobal.dataset.loc[self.dataIx].shape[1]
        parmvals=[]


        init='spectral'
        if dataGlobal.dataset.loc[self.dataIx].shape[0]<=self.dim:
            init='random'

        if self.ffrange == 'kde':
            self.ffrange = ['kde']

        for cutoff in self.ffrange:

            """ Remove columns with low information from data matrix. """

            logging.debug('Features cutoff: {:.3f}'.format(cutoff))

            dataCut, decomposer =self._featuresRemoval(cutoff)


            if self.norm is not None:

                """ Normalize data. """

                logging.debug('Normalize with '+self.norm)

                dataCut=pd.DataFrame(normalize(dataCut, norm=self.norm), index=dataCut.index, columns=dataCut.columns)


            for nn in nnrange:

                """ Project data with UMAP """    

                logging.debug('Number of nearest neighbors: {:d}'.format(nn))

                mapping = umap.UMAP(metric=self.metricM, n_components=self.dim, min_dist=0.0, spread=1, n_neighbors=nn,
                                    n_epochs=self.epochs, learning_rate=self.lr, verbose=False, random_state=self._umapRs,
                                    init=init)
                    
                """if data to be projected only is provided, calculate optimality only on the fit data. """
          
                if self.transform is not None:
                    pj = pd.DataFrame(mapping.fit_transform(dataCut[~dataCut.index.isin(self.transform)]), 
                        index=dataCut[~dataCut.index.isin(self.transform)].index)
                else: 
                    pj = pd.DataFrame(mapping.fit_transform(dataCut), index=dataCut.index) 


                if not pj.isnull().values.any():

                    """ Set cluster_selection_epsilon for HDBSCAN """

                    cse=None        
                    hdbalgo='best'    
                    if self.clusterer=='HDBSCAN':
                        cse=float(self._elbow(pj))
                        if self.metricC=='cosine':
                            hdbalgo='generic'
                            pj=pj.astype(np.float64)

                    parmvals.append([])

                    """ Set clustering parameter range at the first iteration. """

                    #if self.cparmrange=='guess':

                        #self.cparmrange=self._guessEps(pj.sample(np.min([500,pj.shape[0]])))
                        #self.cparmrange=self._guessParm(pj)

                    cparmrange=self.cparmrange
                    if cparmrange=='guess':
                        cparmrange=self._guessParm(pj)
                    #Note: Calculating (H)DBSCAN on a grid of parameters is cheap even with Differential Evolution.

                    for cparm in cparmrange: #self.cparmrange

                        logging.debug('Clustering parameter: {:.5f}'.format(cparm))
                
                        labs = self._findClusters(pj, cparm, cse, hdbalgo)

                        #not 100% sure about this, keep until weights on noise will be added
                        compset=set(labs)
                        compset.discard(-1)

                        if len(compset) > 1:  
                            sil = self.calcScore(pj, labs)
                        else:
                            sil = 0
                        
                        logging.debug('Clustering score: {:.3f}'.format(sil))

                        parmvals[-1].append(sil)

                        if sil > silOpt:
                            cparmOpt = cparm
                            silOpt= sil
                            labsOpt = labs
                            neiOpt = nn
                            pjOpt = pj
                            cutOpt = cutoff
                            mapOpt = mapping 
                            if self.filterfeat in ['variance','MAD']:
                                keepfeat= dataCut.columns
                            if self.filterfeat=='tSVD':
                                decompOpt = decomposer

        """ If an optimal solution was not found. """
        
        if silOpt == -0.0001:
            logging.info('Optimal solution not found!')
            pjOpt = pj
            mapOpt = mapping

        return silOpt, pd.Series(labsOpt, index=pjOpt.index), cparmOpt, neiOpt, pjOpt, cutOpt, mapOpt, keepfeat, decompOpt, parmvals

    def _knn(self, neiOpt, pjOpt, labsOpt, cutoff=None):

        """ KNN classifier (REDUNDANT, TO REMOVE), 
            updates membership assignment for transform-only data.

        Args:
            labsOpt (pandas series): Series with the cluster membership identified for each sample.
            neiOpt (int): Optimal number of nearest neighbors used with UMAP.
            pjOpt (pandas dataframe): Low dimensionality data projection from UMAP.
            
        Returns:
            (pandas series): Series with the updated cluster membership identified for each sample.

        """

        missing=[i for i,x in enumerate(pjOpt.index) if x not in labsOpt.index]
        #print(len(missing))

        neigh=NN(n_neighbors=len(missing)+neiOpt, metric=self.metricM, n_jobs=-1)
        neigh.fit(pjOpt)

        kn=neigh.kneighbors(pjOpt.iloc[missing], return_distance=True)
        kn=np.array([[x,y] for x,y in zip(kn[0],kn[1])])
        mask=~np.isin(kn[:,1],missing)
        #TODO: make prettier
        newk=[[kn[i,0][mask[i]][1:neiOpt+1],kn[i,1][mask[i]][1:neiOpt+1]] for i in range(len(kn))]    

        label_binarizer = LabelBinarizer() 
        label_binarizer.fit(range(-1,len(list(set(labsOpt)))+1))
        tmplab = pd.DataFrame(label_binarizer.transform(labsOpt))
        tmplab.columns=range(-1,len(list(set(labsOpt)))+1)
        tmplab.drop(tmplab.columns.values[tmplab.sum() <= 0], axis=1, inplace=True)    
        tmplab.index=labsOpt.index 

        valals=[]   
        for k in np.arange(len(newk)):
            vals=tmplab.loc[pjOpt.iloc[newk[k][1]].index].apply(lambda x: x/newk[k][0], axis=0)[1:]
            valals.append((vals.sum(axis=0)/vals.sum().sum()).values)

        knnlabs=pd.DataFrame(valals, index=pjOpt.index[missing], columns=tmplab.columns)

        if cutoff!=None:
            knnlabs[knnlabs<cutoff]==0
            knnlabs[-1]=cutoff-.00001

        labsNew=knnlabs.idxmax(axis=1)
        #labsNew.index=missing
        #print('labsNew',labsNew)

        return pd.concat([labsOpt,labsNew], axis=0)


    def _optimizeParams(self):

        """ Wrapper function for the parameters optimization.

        Returns:
            silOpt (float): Silhoutte score corresponding to the best set of parameters.
            labsOpt (pandas series): Series with the cluster membership identified for each sample.
            cparmOpt (float): Optimal clusters identification parameter value found.
            numClusOpt (int):  Total number of clusters determined by the search.
            neiOpt (int): Optimal number of nearest neighbors used with UMAP.
            pjOpt (pandas dataframe): Low dimensionality data projection from UMAP.
            cutOpt (float): Optimal cutoff value used for the features removal step.
            keepfeat (pandas index): Set of genes kept after low variance/MAD removal, nan if tSVD.
            decompOpt (tsvd object): trained tsvd instance, None if 'variance'/'MAD'.
            reassigned (float): Percentage of points forecefully assigned to a class 
                if outliers='reassign'.
            parmvals (list of float): List of all scores evaluated. 

        """
   
        logging.info('Dimensionality of the target space: {:d}'.format(self.dim))
        logging.info('Samples #: {:d}'.format(dataGlobal.dataset.loc[self.dataIx].shape[0]))

        if self.transform is not None:
            logging.info('Transform-only Samples #: {:d}'.format(len(self.transform)))


        numpoints=dataGlobal.dataset.loc[self.dataIx].shape[0]
        if (self.transform is not None):
            numpoints-=len(self.transform)

        if self.neirange == 'logspace':
            minbound=np.log10(np.sqrt(numpoints*self.neifactor))
            maxbound=np.log10(numpoints*self.neifactor)
            if self.neifactor>=1:
                maxbound=np.log10(numpoints-1)
            nnrange = sorted([int(x) for x in np.logspace(
                minbound, maxbound, num=self.neipoints[0])])
        elif self.neirange == 'sqrt':
            nnrange = [int(np.sqrt(numpoints*self.neifactor))]
        else:
            nnrange=self.neirange

        """ Run Optimizer. """

        if self.optimizer=='grid':

            """ Grid Search """
            logging.info('Running Grid Search...')

            silOpt, labsOpt, cparmOpt, neiOpt, pjOpt, cutOpt, mapOpt, keepfeat, decompOpt, parmvals = self._runGridInstances(nnrange)
            
            logging.info('Done!')

        elif self.optimizer=='de':

            """ Differential Evolution """

            logging.info('Running Differential Evolution...')
            
            #Note: this works as monodimensional DE, but may be slightly inefficient
            bounds=[(np.min(self.ffrange),np.max(self.ffrange)),(np.min(nnrange),np.max(nnrange))]
            silOpt, labsOpt, cparmOpt, neiOpt, pjOpt, cutOpt, mapOpt, keepfeat, decompOpt, parmvals = \
            de._differentialEvolution(self._objectiveFunction, bounds, maxiter = self.deiter[0], popsize = self.depop[0], integers=[False, True])

            #DEPRECATED
            #bestParam = de._differentialEvolution(self._objectiveFunction, bounds, maxiter = self.deiter, popsize = self.depop, integers=[False, True])
            #cutOpt=bestParam[0]
            #neiOpt=int(bestParam[1])
            #silOpt, labsOpt, cparmOpt, pjOpt, mapOpt, keepfeat = self._runSingleInstance(cutOpt,neiOpt)

            logging.info('Done!')

        else:
            sys.exit('ERROR: optimizer not recognized')

        """ If data to be projected only is provided, apply projection. """

        if (self.transform is not None):

            #A bit redundant, try to clean up
            if self.filterfeat in ['variance','MAD']:
                transdata=dataGlobal.dataset.loc[self.dataIx][keepfeat].loc[self.transform]
            elif self.filterfeat=='tSVD':
                transdata=self._featuresRemoval(cutOpt).loc[self.transform]

            pjOpt=pd.concat([pjOpt, pd.DataFrame(mapOpt.transform(transdata), 
                index=self.transform)], axis=0)

            logging.debug('Transform-only data found at this level: membership will be assigned with KNN')

            """ Assign cluster membership with k-nearest neighbors. """

            labsOpt=self._knn(neiOpt, pjOpt, labsOpt) 
        
            #alternative to nearest neighbors: DEPRECATED
            #labsOpt = DBSCAN(eps=epsOpt, min_samples=10, metric=self.metricC, n_jobs=-1, leaf_size=15).fit_predict(pjOpt)
            #labsOpt = pd.Series(labsOpt)
            

        """ Dealing with discarded points if outliers!='ignore' 
            applies only if there's more than one cluster identified
            and if at least 10% but less than 90% of the samples have been discarded. 
        """

        reassigned = 0.0
        if (self.outliers=='reassign') and (-1 in labsOpt.values) and (labsOpt.unique().shape[0]>2):
            if labsOpt.value_counts().to_dict()[-1]>labsOpt.shape[0]*.1 and labsOpt.value_counts().to_dict()[-1]<labsOpt.shape[0]*.9:

                labsOut=labsOpt[labsOpt!=-1]
                reassigned = (labsOpt.shape[0]-labsOut.shape[0])*1.0/labsOpt.shape[0]
                labsOpt=self._knn(neiOpt, pjOpt, labsOut, cutoff=.5).loc[labsOpt.index]

        

        numClusOpt = len(set(labsOpt)) - (1 if -1 in labsOpt else 0)

        logging.info('\n=========== Optimization Results '+self._name+' ===========\n'+\
            'Features # Cutoff: {:.5f}'.format(cutOpt)+'\n'+\
            'Nearest neighbors #: {:d}'.format(neiOpt)+'\n'+\
            'Clusters identification parameter: {:.5f}'.format(cparmOpt)+'\n'+\
            'Clusters #: {:d}'.format(numClusOpt)+'\n')

        return silOpt, labsOpt, cparmOpt, numClusOpt, neiOpt, pjOpt, cutOpt, mapOpt, keepfeat, decompOpt, reassigned, parmvals


    def recurse(self):

        """ Recursively clusters the input data, by first optimizing the parameters, binirizing the resulting labels, plotting and repeating. """

        self._depth += 1
        
        if (self._levelCheck()):
            return

        if dataGlobal.dataset.loc[self.dataIx].shape[0] < self.popcut:         
            logging.info('Population too small!')
            self.clusOpt = None
            return


        minimum, clusTmp, chosen, nClu, nNei, pj, cut, chomap, keepfeat, decompOpt, reassigned, parmvals = self._optimizeParams()

        """ Save cluster best parameters to table."""

        vals = ['cluster ' + self._name, dataGlobal.dataset.loc[self.dataIx].shape[0], nClu, self.dim, minimum, nNei, chosen, cut, self.metricM, self.metricC, reassigned] 
               
        with open(os.path.join(self.outpath,'raccoonData/paramdata.csv'), 'a') as file:
                writer = csv.writer(file)
                writer.writerow(vals)
                file.close()

        """ Save intermediate UMAP data to be able to re-access specific clusters (expensive, only if debug True). """

        if (self.savemap == True):
            with open(os.path.join(self.outpath,'raccoonData/'+self._name+'.pkl'), 'wb') as file:
                if self.filterfeat in ['variance','MAD']:
                    pickle.dump([keepfeat,chomap], file)
                elif self.filterfeat=='tSVD':
                    pickle.dump([decompOpt,chomap], file)
                file.close()
            pj.to_hdf(os.path.join(self.outpath,'raccoonData/'+self._name+'.h5'), key='proj')

        if nClu < 2:
            logging.info('Optimal solution has only one cluster!')
            self.clusOpt = None
            return

        """ Plotting. """        

        self._plot(nNei, pj, cut, keepfeat, decompOpt, clusTmp, parmvals)


        """ Binarize data. """

        clusTmp = self._binarize(clusTmp)

        """ Dig within each subcluster and repeat. """

        for l in list(clusTmp.columns):

            selNew = dataGlobal.dataset.loc[self.dataIx].loc[clusTmp[clusTmp[l]==1].index]
 
            if (self.lab is not None):
                labNew = self.lab.loc[clusTmp[clusTmp[l]==1].index]
            else:
                labNew = None

            logging.info('Going deeper within Cluster # ' +
                  str(l)+' [depth: {:d}'.format(self._depth)+']')

            if (self.transform is not None):
                indices = list(selNew.index.values)
                to_transform = [x for x in indices if x in self.transform]
                if not to_transform:
                    to_transform = None
            else:
                to_transform = None

            """ Move along the list of parameters to change granularity """
            if self.optimizer=='grid' and len(self.ffpoints)>1:
                self.ffpoints=self.ffpoints[1:]
                logging.info('Parameters granilarity change ' +
                '[features filter: {:d}'.format((self.ffpoints[0]))+']')
            if self.optimizer=='grid' and len(self.neipoints)>1:
                self.neipoints=self.neipoints[1:]
                logging.info('Parameters granilarity change ' +
                '[nearest neighbours: {:d}'.format((self.neipoints[0]))+']')
            if self.optimizer=='de' and len(self.depop)>1:
                self.depop=self.depop[1:]
                logging.info('Parameters granilarity change ' +
                '[DE population: {:d}'.format(int(self.depop[0]))+']')
            if self.optimizer=='de' and len(self.deiter)>1:
                self.deiter=self.deiter[1:]                
                logging.info('Parameters granilarity change ' +
                '[DE iterations: {:d}'.format(int(self.deiter[0]))+']')

            deep = recursiveClustering(selNew.index, lab=labNew, transform=to_transform, dim=self.dim, epochs=self.epochs, lr=self.lr, 
                                      neirange=self.neirange, neipoints=self.neipoints, metricM=self.metricM, metricC=self.metricC, 
                                      popcut=self.popcut, filterfeat=self.filterfeat, ffrange=self.ffrange, ffpoints=self.ffpoints, 
                                      optimizer=self.optimizer, depop=self.depop, deiter=self.deiter, score=self.score, norm=self.norm, 
                                      clusterer=self.clusterer, cparmrange=self.cparmrange, minclusize=self.minclusize, outliers=self.outliers, 
                                      name=str(l), debug=self.debug, maxdepth=self.maxdepth, savemap=self.savemap, RPD=self.RPD, 
                                      outpath=self.outpath, depth=self._depth, cores=self._cores, _user=False)

            deep.recurse() 


            if deep.clusOpt is not None:
      
                clusTmp = pd.concat([clusTmp, deep.clusOpt], axis=1, join='outer')
                clusTmp = clusTmp.fillna(0)

                cols = list(clusTmp.columns.values)
                for col in cols:
                    clusTmp[col] = clusTmp[col].astype(int)


        self.clusOpt = clusTmp


def run(data, **kwargs):

    """ Wrapper function to setup, create a recursiveClustering object, run the recursion and logging in serial.

        Args:
            data (pandas dataframe): dataframe with sampels as rows and features as columns.
            **kwargs: keyword arguments for recursiveClustering.

        Returns:
            clusOpt (pandas dataframe): one-hot-encoded clusters membership of data.
    """

    if 'outpath' not in kwargs or kwargs['outpath'] is None:
        kwargs['outpath']=os.getcwd()
    if 'RPD' not in kwargs:
        kwargs['RPD']=False

    """ Setup folders and files, remove old data if present.  """

    functions.setup(kwargs['outpath'], kwargs['RPD'])

    """ Run recursive clustering algorithm. """

    obj = recursiveClustering(data, **kwargs) 
    obj.recurse()

    """ Save the assignment to disk. """

    obj.clusTmp.to_hdf(os.path.join(kwargs['outpath'],'finalOutput.h5'),key='df')
    
    """ Log the total runtime and memory usage. """

    logging.info('=========== Final Clustering Results ===========')
    if obj.clusOpt is not None:
        logging.info('A total of {:d} clusters were found'.format(len(obj.clusOpt.columns)))
    else:
        logging.info('No clusters found! Try changing the search parameters')
    logging.info('Total time of the operation: {:.3f} seconds'.format((time.time() - start_time)))
    logging.info(psutil.virtual_memory())
  
    return obj.clusOpt


if __name__ == "__main__":

    pass
