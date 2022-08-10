"""
Clustering classes and functions for RACCOON
F. Comitani     @2018-2022
A. Maheshwari   @2019
"""

import os
import sys
#from shutil import copyfile

import csv
import pickle

import logging
DEBUG_R = 15

""" Suppress UMAP and numpy warnings. """

import numba
import warnings

warnings.filterwarnings(
    "ignore",
    message="n_neighbors is larger than the dataset size; truncating to")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=numba.errors.NumbaPerformanceWarning)

from math import nan
import random

from raccoon import interface
from raccoon.classification import local_KNN

import raccoon.utils.plots as plotting
import raccoon.utils.functions as functions
import raccoon.utils.trees as trees
import raccoon.optim.de as de 
import raccoon.optim.tpe as tpe 
import raccoon.utils.classes as classes
from raccoon.utils.option import OptionalImports

""" Search for optional libraries.  """

try:
    import hdbscan as HDBSCAN
    OptionalImports.hdbscan = True
except ImportError:
    pass

try:
    import feather
    OptionalImports.feather = True
except ImportError:
    pass

__version__ = "0.4.0"

class DataGlobal:

    """ Static container for the input data to be filled
        by the user at the first iteration. """

    dataset = None
    labels = None


class IterativeClustering:

    """ To perform top-down iterative  clustering on a samples x features matrix. """

    def __init__(self, data, lab=None, transform=None, supervised=False, 
                supervised_weight=0.5, dim=2, epochs=5000, lr=0.05, 
                nei_range='logspace', nei_points=25, nei_factor=1.0,
                neicap=100, skip_equal_dim=True, skip_dimred=False,
                metric_map='cosine', metric_clu='euclidean', pop_cut=50,
                filter_feat='variance', ffrange='logspace', ffpoints=25,
                optimizer='grid', search_candid=10, search_iter=10,
                tpe_patience=5, score='silhouette', baseline=-1e-5,
                norm=None, dyn_mesh=False, max_mesh=20, min_mesh=4,
                clu_algo='SNN', cparm_range='guess', min_sam_dbscan=None, 
                outliers='ignore', noise_ratio=.3, min_csize=None,
                name='0', debug=False, max_depth=None,
                save_map=True, RPD=False,
                out_path="", depth=0, chk=False, 
                gpu=False, _user=True):
        """ Initialize the the class.

        Args:
            data (matrix, pandas dataframe or pandas index): if first call (_user==True), input data
                in pandas dataframe-compatible format (samples as row, features as columns),
                otherwise index of samples to carry downstream during the iteration calls.
            lab (list, array or pandas series): list of labels corresponding to each sample
                (for plotting only).
            transform (list of Pandas DataFrame indices): list of indices of the samples in the
                initial matrix that should be transformed-only
                and not used for training the dimensionality reduction map.
            supervised (bool): if true, use labels for supervised dimensionality reduction with
                UMAP (default False, works only if lab !=None).
            supervised_weight (float): how much weight is given to the labels in supervised UMAP
                (default 0.5).
            if true, use labels for supervised dimensionality reduction with
                UMAP (default False, works only if lab !=None).
            dim (integer): number of dimensions of the target projection (default 2).
            epochs (integer): number of UMAP epochs (default 5000).
            lr (float): UMAP learning rate (default 0.05).
            nei_range (array, list of integers or string or function): list of nearest neighbors values to be
                used in the search;
                if 'logspace' take an adaptive range based on the dataset size at each iteration
                with logarithmic spacing (reccomended),
                if a function is provided it will be used to define the neighbors range at each step
                (see the manual for more details).
            nei_points (int or list of int): number of grid points for the neighbors search,
                if list, each value will be subsequently used at the next iteration until
                all values are exhausted,
                (works only with optimizer='grid' and nei_range='logspace' default 25).
            nei_factor (float): scaling factor for 'logspace' and 'sqrt' selections in nei_range
            neicap (int): maximum number of neighbors (reccomended with low-memory systems,
                default 100).
            skip_equal_dim (bool): if True, whenever the target dimensionality corresponds
                to the dimensionality of the input data, the dimensionality reduction step will
                be skipped (saves time, default True).
            skip_dimred (bool): if True, skip the non-linear dimensionality reduction step 
                (default False).
            metric_map (string): metric to be used in UMAP distance calculations (default cosine).
            metric_clu (string): metric to be used in clusters identification and clustering score
                calculations (default euclidean)
                Warning: 'cosine' does not work with HDBSCAN, normalize to 'l2' and use 'euclidean'
                instead.
            pop_cut (integer): minimum number of samples for a cluster to be considered valid, if a
                cluster is found with a lower population than this threshold, it will not be further
                explored (default 50).
            filter_feat (string): set the method to filter features in preprocessing;
                if 'variance' remove low variance genes
                if 'MAD' remove low median absolute deviation genes
                if 'correlation' remove correlated genes
                if 'tSVD' use truncated single value decomposition (LSA)
            ffrange (array, list or string): if filter_feat=='variance'/'MAD'/'correlation',
                percentage values for the low-variance/correlation removal cufoff search;
                if 'logspace' (default) take a range between .3 and .9 with logarithmic spacing
                (reccomended, will take the extremes if optimizer=='de');
                if 'kde' kernel density estimation will be used to find a single optimal
                low-variance cutoff (not compatible with optimizer=='de')
                if filter_feat=='tSVD', values for the number of output compontents search;
                if 'logspace' (default) take a range between number of features times .3 and
                .9 with logarithmic spacing
                (reccomended, will take the extremes if optimizer=='de')
            ffpoins (int or list of int): number of grid points for the feature removal cutoff search
                if list, each value will be subsequently used at the next iteration until
                all values are exhausted,
                (works only with ffrange='logspace', default 25).
            optimizer (string): choice of parameters optimizer, can be either 'grid' for grid search,
                'de' for differential evolution, 'tpe' for Tree-structured Parzen Estimators with Optuna,
                or 'auto' for automatic (default is 'grid').
                Automatic will chose between grid search and DE depending on the number of
                search points (de if >25), works only if dyn_mesh is True.
            search_candid (int or list of int): size of the candidate solutions population in
                DE or TPE.
                If list, each value will be subsequently used at the next iteration until
                all values are exhausted (this last option works only with optimizer='de' and 'tpe', 
                default 10).
            search_iter (int or list of int): maximum number of iterations of differential evolution.
                If list, each value will be subsequently used at the next iteration until
                all values are exhausted (works only with optimizer='de', default 10).
            tpe_patience (int): number of tpe iteractions below the tolerance before interrupting 
                the search.
            score (string or function): objective function of the optimization, to be provided as
                a string (currently only 'dunn' and 'silhouette' are available, default 'silhouette').
                Alternatively, a scoring function can be provided, it must take a feature array,
                an array-like list of labels and a metric, in the same format as 
                sklearn.metrics.silhouette_score.
            baseline (float): baseline score. Candidate parameters below this score will be 
                automatically excluded (defaul -1e5).
            norm (string): normalization factor before dimensionality reduction (default None),
                not needed if metric_map is cosine
                if None, don't normalize.
            dyn_mesh (bool): if true, adapt the number of mesh points (candidates and iteration in DE, 
                candidates in tpe) to the population, overrides nei_points, search_candid, 
                search_iter and ffpoints (default False).
            max_mesh (int): maximum number of points for the dyn_mesh option (hit at 10000 samples,
                default 20), this is a single dimension, the actuall mesh will contain n*n points.
            min_mesh (int): minimum number of points for the dyn_mesh option (hit at 50 samples,
                default 4, must be >3 if optimizer='de'),
                this is a single dimension, the actuall mesh will contain n*n points.
            clu_algo (string): selects which algorithm to use for clusters identification.
                Choose among 'DBSCAN', 'SNN' (Shared Nearest Neighbors DBSCAN, default),  
                'HDBSCAN', or 'louvain' (Louvain community detection with SNN).
            cparm_range (array, list) or string: clusters identification parameter range to
                be explored (default 'guess').
                When 'DBSCAN' this corresponds to epsilon (if 'guess' attempts to identify it
                by the elbow method);
                When 'HDBSCAN' this corresponds to the minimum number of samples required by
                the clusters (if 'guess' adapts it on the dataset population).
            min_sam_dbscan (int): minimum number of samples to define a core used in DBSCAN and HDBSCAN.
                if None, set 2*target_dim (default None)
                (default is 10).
            outliers (string): selects how to deal with outlier points in the clusters assignment
                if 'ignore' discard them
                if 'reassign' try to assign them to other clusters with knn if more than 10%
                of the total population was flagged.
            noise_ratio (float): maximum percentage cutoff of samples that can be labelled as noise
                 before discarding the result (relevant only for clustering algorithms that label border 
                 points as noise, default .3).
            min_csize (int): Minimum population size of clusters. If None, keep all clusters,
                else, clusters below this threshold will be discarded as soon as they are 
                identified (default None).
            name (string): name of current clustering level (should be left as default, '0',
                unless continuing from a previous run).
            debug (boolean): specifies whether algorithm is run in debug mode (default is False).
            max_depth (int): Specify the maximum number of search iterations, if None (default),
                keep going while possible.
                0 stops the algorithm immediately, 1 stops it after the first level.
            save_map (boolean): if active, saves the trained maps to disk (default is True).
                Needed to run the k-NN classifier.
            RPD (boolean): specifies whether to save RPD distributions for each cluster
                (default is False). Warning: this option is unstable and not reccomended.
            out_path (string): path to the location where outputs will be saved (default,
                save to the current folder).
            depth (integer): current depth of search (should be left as default
                0, unless continuing from a previous run).
            chk (bool): save checkpoints (default False, reccomended for big jobs).
            gpu (bool): Activate GPU version (requires RAPIDS).
            _user (bool): Boolean switch to distinguish initial user input versus iteration
                calls, do not change.
        """

        """ Set up for CPU or GPU run. """

        self.gpu = gpu

        if self.gpu:
            try:
                self.interface = interface.InterfaceGPU()
            except BaseException:
                warnings.warn("No RAPIDS found, running on CPU instead.")
                self.gpu = False

        if not self.gpu:
            self.interface = interface.InterfaceCPU()

        if _user:

            if not isinstance(data, self.interface.df.DataFrame):
                try:
                    data = self.interface.df.DataFrame(data)
                except BaseException:
                    if self.gpu:
                        try:
                            data = self.interface.df.from_pandas(data.astype(float))
                        except BaseException:
                            pass
                    print("Unexpected error: ", sys.exc_info()[0])
                    print("Input data should be in a format that can be translated "+\
                           "to pandas/cuDF dataframe!")
                    raise

            DataGlobal.dataset = data
            data = data.index

            if lab is not None and not isinstance(
                    lab, self.interface.df.Series):
                try:
                    lab = self.interface.df.Series(lab)
                except BaseException:
                    print("Unexpected error: ", sys.exc_info()[0])
                    print("Labels data should be in a format that can be translated "+\
                          "to pandas series!")
                    raise
                try:
                    lab.index = data
                except BaseException:
                    print("Unexpected error: ", sys.exc_info()[0])
                    print("Shape of labels data should be consistent with the shape of data!")
                    raise
            
            DataGlobal.labels = lab

        self.data_ix = data
        self.transform = transform

        self.supervised = supervised
        if self.supervised and DataGlobal.labels is None:
            warnings.warn("Labels need to be provided for supervised dimensionality "+\
                           "reduction, setting supervised to False.")
            self.supervised = False

        self.super_w = supervised_weight
        self.optimizer = optimizer
        # Keep track of the original optimizer if changed
        self.optimtrue = optimizer

        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.nei_range = nei_range
        self.nei_points = nei_points
        self.nei_factor = nei_factor
        self.neicap = neicap
        self.skip_equal_dim = skip_equal_dim
        self.skip_dimred = skip_dimred
        self.metric_map = metric_map
        self.metric_clu = metric_clu
        self.mparams = {}
        self.pop_cut = pop_cut
        self.filter_feat = filter_feat
        self.ffrange = ffrange
        self.ffpoints = ffpoints
        self.debug = debug
        self.save_map = save_map
        self.max_depth = max_depth
        self.min_csize = min_csize
        self.RPD = RPD
        self.search_candid = search_candid
        self.search_iter = search_iter
        self.out_path = out_path
        self.clus_opt = None
        self._name = name
        self._depth = depth
        self.tpe_patience = tpe_patience
        self.score = score
        self.baseline = baseline
        self.norm = norm
        self.dyn_mesh = dyn_mesh
        self.max_mesh = max_mesh
        self.min_mesh = min_mesh

        self.clu_algo = clu_algo
        self.cparm_range = cparm_range
        self.min_sam_dbscan = min_sam_dbscan
        self.outliers = outliers
        self.noise_ratio = noise_ratio

        self.chk = chk

        if self.min_sam_dbscan is None:
            # Sander et al. 1998
            self.min_sam_dbscan = self.dim * 2

        """ CPU vs GPU methods check. """

        if self.gpu:
            if self.clu_algo not in  ['DBSCAN']:
                warnings.warn("For now, Only DBSCAN is available with RAPIDS, "+\
                               "setting clu_algo as DBSCAN! Please check the github repo for updates.")
                self.clu_algo = 'DBSCAN'

            if self.metric_map != 'euclidean' or self.metric_clu != 'euclidean':
                warnings.warn("Only euclidean is available with RAPIDS, setting metrics "+\
                               "as euclidean!")
                self.metric_map = 'euclidean'
                self.metric_clu = 'euclidean'

        """ Checks on optimizer choice. """

        if self.optimizer not in ['grid', 'de', 'auto', 'tpe']:
            sys.exit('ERROR: Optimizer must be either \'grid\' for Grid Search, \'de\' '+\
                      'for Differential Evolution, \'tpe\' for Tree-structured Parzen Estimators '+\
                      'optimization with Optuna or \'auto\' for automatic selection.')

        if self.optimizer == 'de' and self.ffrange == 'kde':
            sys.exit('ERROR: KDE estimation of the low variance/MAD removal cutoff is '+\
                      'not compatible with Differential Evolution.')

        if self.filter_feat not in ['variance', 'MAD', 'correlation', 'tSVD']:
            sys.exit('ERROR: Features filter must be either \'variance\' for low-variance removal, '+\
                      '\'MAD\' for low-MAD removal, \'correlation\' for correlation removal, '+\
                      'or \'tSVD\' for truncated SVD.')

        if self.optimizer == 'auto' and dyn_mesh == False:
            self.optimizer = 'grid'
            warnings.warn('Optimizer \'auto\' works only if dynamic mesh is active, '+\
                           'falling to Grid Search')

        """ Checks optional modules. """

        if self.clu_algo == 'HDBSCAN' and not OptionalImports.hdbscan:

            warnings.warn("HDBSCAN not found, setting clu_algo as DBSCAN!")
            self.clu_algo = 'DBSCAN'

        """ Evaluate parameters granularity options. """

        if self.dyn_mesh:

            meshpoints = self.interface.num.rint(
                ((self.max_mesh - 1) * functions.sigmoid(
                    self.interface.num.log(
                        self.data_ix.shape[0]),
                    self.interface,
                    a=self.interface.num.log(500),
                    b=1))) + (
                self.min_mesh)

            if self.optimizer == 'auto':
                self.optimizer = 'de' if meshpoints**2 > 25 else 'grid'

            if self.optimizer == 'grid':

                self.nei_points = meshpoints
                self.ffpoints = meshpoints

            elif self.optimizer in ['de', 'tpe'] :

                if self.min_mesh <= 3:
                    self.min_mesh = 4
                    meshpoints = self.interface.num.rint(
                        ((self.max_mesh - 1) * functions.sigmoid(
                            self.interface.num.log(
                                self.data_ix.shape[0]),
                            self.interface,
                            a=self.interface.num.log(500),
                            b=1))) + (
                        self.min_mesh)

                self.search_candid = meshpoints
                self.search_iter = meshpoints
                
                if self.optimizer == 'tpe':
                    self.search_candid = self.search_candid**2

        try:
            self.nei_points = [int(x) for x in self.nei_points] \
                                if isinstance(self.nei_points, list) \
                                else [int(self.nei_points)]

        except BaseException:
            sys.exit('ERROR: nei_points must be an integer or a list of integers')
            raise

        if self.optimizer in ['de','tpe']:
            try:
                self.search_candid = [int(x) for x in self.search_candid] \
                                        if isinstance(self.search_candid, list) \
                                        else [int(self.search_candid)]

            except BaseException:
                sys.exit('ERROR: search_candid must be an integer or a list of integers')
                raise
            if self.ffrange == 'logspace':
                if self.filter_feat in ['variance', 'MAD', 'correlation']:
                    self.ffrange = [0.3, 0.9]
                if self.filter_feat == 'tSVD':
                    self.ffrange = [int(min([50,
                                             DataGlobal.dataset.loc[self.data_ix].shape[1] * 0.3])),
                                    int(max([1,
                                             DataGlobal.dataset.loc[self.data_ix].shape[1] * 0.9]))]
                    self.ffrange = [x for x in self.ffrange if x>0]

        if self.optimizer == 'de':
            try:
                self.search_iter = [int(x) for x in self.search_iter] \
                                        if isinstance(self.search_iter, list) \
                                        else [int(self.search_iter)] 

            except BaseException:
                sys.exit('ERROR: search_iter must be an integer or a list of integers')
                raise
            
        if self.optimizer == 'grid':
            try:
                self.ffpoints = [int(x) for x in self.ffpoints] \
                                    if isinstance(self.ffpoints, list) \
                                    else [int(self.ffpoints)]

            except BaseException:
                sys.exit('ERROR: ffpoints must be an integer or a list of integers')
                raise
            if self.ffrange == 'logspace':
                if self.filter_feat in ['variance', 'MAD', 'correlation']:
                    self.ffrange = sorted([float(x) for x in self.interface.num.logspace(
                        self.interface.num.log10(0.3),
                        self.interface.num.log10(0.9),
                        num=self.ffpoints[0])])
                if self.filter_feat == 'tSVD':
                    self.ffrange = sorted([int(x) for x in self.interface.num.logspace(
                        self.interface.num.log10(min([50,
                            DataGlobal.dataset.loc[self.data_ix].shape[1] * 0.3])),
                        self.interface.num.log10(max([1,
                            DataGlobal.dataset.loc[self.data_ix].shape[1] * 0.9])),
                        num=self.ffpoints[0])])
                    self.ffrange = [x for x in self.ffrange if x>0]

        """ Setup logging. """

        if self.debug:
            logging.addLevelName(DEBUG_R, 'DEBUG_R')
            logging.getLogger().setLevel(DEBUG_R)
            self._seed = 32
        else:
            logging.getLogger().setLevel(logging.INFO)
            self._seed = random.randint(0,999999)
        
        self._umap_rs = self._seed
        logging.info(
            "Random seed is: {:d}".format(int(self._seed)))

        scorename = 'user defined' if hasattr(self.score, '__call__') \
                        else self.score 

        logging.info(
            "Scoring function is: "+scorename)

    def _features_removal(self, cutoff):
        """ Either remove features with low variance/MAD, or high correlation
        from dataset according to a specified threshold (cutoff)
        or apply truncated SVD to reduce features to a certain number (cutoff). 

        Args:
             cutoff (string or float): if filter_feat=='variance'/'MAD'/'correlation',
                percentage value for the low-variance/MAD/high-correlation removal cufoff,
                if 'kde' kernel density estimation will be used to find a single optimal
                low-variance/MAD/high-correlation cutoff;
                if filter_feat=='tSVD', dimensionality of the output data.

        Returns:
            (pandas dataframe): reduced-dimensionality input data.
            (tsvd object): trained tsvd instance, None if 'variance'/'MAD'/'correlation'.

        """

        if self.filter_feat == 'tSVD':

            if int(cutoff) >= DataGlobal.dataset.shape[1]:
                logging.info(
                    "{:d} features cutoff >= dimensionality of input data,"+\
                    " t-SVD will be skipped".format(int(cutoff)))
    
                return DataGlobal.dataset.loc[self.data_ix], None

            logging.info(
                "Applying t-SVD with {:d} features".format(int(cutoff)))

            decomposer = self.interface.decompose(n_components=int(cutoff))

            """ Add conditional to apply the cut only on those samples used for training the map. """

            # Ugly but hopefully more memory efficient
            # csr_matrix not compatible with RAPIDS
            # if self.transform is not None:
            #    decomposer.fit(csr_matrix(DataGlobal.dataset.loc[self.data_ix][~DataGlobal.dataset.loc[self.data_ix].index.isin(self.transform)].values))
            #    return self.interface.df.DataFrame(decomposer.transform(csr_matrix(DataGlobal.dataset.loc[self.data_ix].values)), index=DataGlobal.dataset.loc[self.data_ix].index), decomposer
            # else:
            # return
            # self.interface.df.DataFrame(decomposer.fit_transform(csr_matrix(DataGlobal.dataset.loc[self.data_ix].values)),
            # index=DataGlobal.dataset.loc[self.data_ix].index), decomposer

            if self.transform is not None:
                decomposer.fit(DataGlobal.dataset.loc[self.data_ix]
                    [~DataGlobal.dataset.loc[self.data_ix].index.isin(
                        self.transform)].values)
                return self.interface.df.DataFrame(decomposer.transform(
                    DataGlobal.dataset.loc[self.data_ix].values),
                    index=DataGlobal.dataset.loc[self.data_ix].index), decomposer

            return self.interface.df.DataFrame(decomposer.fit_transform(
                DataGlobal.dataset.loc[self.data_ix].values),
                index=DataGlobal.dataset.loc[self.data_ix].index), decomposer

        """ Add conditional to apply the cut only on those samples used for training the map. """

        if self.transform is not None:
            new_data = DataGlobal.dataset.loc[self.data_ix]\
                [~DataGlobal.dataset.loc[self.data_ix].index.isin(
                self.transform)]
        else:
            new_data = DataGlobal.dataset.loc[self.data_ix]

        if self.filter_feat in ['variance', 'MAD']:

            if cutoff == 'kde':
                new_data = functions._drop_min_KDE(
                    new_data, self.interface, type=self.filter_feat)
            elif cutoff < 1:
                new_data = functions._near_zero_var_drop(
                    new_data, self.interface, thresh=cutoff, type=self.filter_feat)

            logging.log(DEBUG_R, "Dropped Features #: " + '{:1.0f}'.format(
                DataGlobal.dataset.loc[self.data_ix].shape[1] - new_data.shape[1]))

            # Extra passage needed in case the transform data cut was
            # applied
            return DataGlobal.dataset.loc[self.data_ix][new_data.columns], None

        elif self.filter_feat == 'correlation':

            if cutoff < 1:
                new_data = functions._drop_collinear(
                    new_data, self.interface, thresh=cutoff)

            logging.log(DEBUG_R, "Dropped Features #: " + '{:1.0f}'.format(
                DataGlobal.dataset.loc[self.data_ix].shape[1] - new_data.shape[1]))

            # Extra passage needed in case the transform data cut was
            # applied
            return DataGlobal.dataset.loc[self.data_ix][new_data.columns], None

        else:

            sys.exit('ERROR: Oops, something went really wrong! Make sure filter_feat \
                        is properly set.\nIf you see this error please contact us on GitHub!')

    
    def _level_check(self):
        """ Stop the iterative search if a given max_depth parameter has been reached. """

        if self.max_depth is not None:
            #self.max_depth -= 1
            #if self.max_depth == -1:
            if self.max_depth <= self._depth:
                #self.clus_opt = None
                return True
        return False

    def _plot(self, n_nei, proj, cut_opt,
              keepfeat, decomposer, clus_opt, scoreslist):
        """ Produce a number of plots to visualize the clustering outcome at each stage of 
        the iterative search.

        Args:
            n_nei (integer): optimal number of nearest neighbors (used in UMAP)
                that was found through grid search.
            proj (pandas dataframe of floats): optimal reduced dimensionality data matrix.
            cut_opt (int or float): optimal features removal cutoff.
            keepfeat (pandas index): set of genes kept after low /MAD removal, nan if tSVD.
            decomposer (tsvd object): trained tsvd instance.
            clus_opt (pandas series): cluster membership series.
            scoreslist (list of float): list of all scores evaluated and their parameters.

        """

        """ Plot the score optimization surface. """

        if len(set(scoreslist[0])) > 1 and len(set(scoreslist[1])) > 1:
            plotting._plot_score_surf(scoreslist,
                                    (cut_opt, n_nei),
                                    'scores_' + self._name,
                                    self.out_path)
        elif len(set(scoreslist[0])) > 1 and len(set(scoreslist[1])) == 1:
            plotting._plot_score([scoreslist[0],
                                 scoreslist[2]],
                                cut_opt,
                                'Features filter',
                                'scores_' + self._name,
                                self.out_path)
        elif len(set(scoreslist[1])) > 1 and len(set(scoreslist[0])) == 1:
            plotting._plot_score([scoreslist[1],
                                 scoreslist[2]],
                                n_nei,
                                'Nearest neighbors',
                                'scores_' + self._name,
                                self.out_path)

        """ Plot the Relative Pairwise Distance (RPD) distributions. """

        if self.RPD:
            #WARNING: unstable
            try:
                functions._calc_RPD(proj, clus_opt, self.interface,
                    True, self._name, self.out_path)
            except BaseException:
                logging.warning('RPD failed at step: ' + self._name)

        if self.filter_feat in ['variance','MAD'] \
            and isinstance(keepfeat,self.interface.df.Index):
            selcut = DataGlobal.dataset.loc[self.data_ix][keepfeat]

            """ Plots distribution of variance/MAD and low-variance/MAD genes cutoff. """

            plotting._plot_cut(self.interface.get_value(DataGlobal.dataset.loc[self.data_ix],
                              pandas=True),
                              self.interface.get_value(selcut, pandas=True),
                              'cut_' + self._name, self.out_path)

        elif self.filter_feat == 'tSVD' and decomposer is not None:
            # scr_matrix not compatible with RAPIDS
            #selcut=self.interface.df.DataFrame(decomposer.transform(csr_matrix(DataGlobal.dataset.loc[self.data_ix].values)), index=DataGlobal.dataset.loc[self.data_ix].index)
            selcut = self.interface.df.DataFrame(decomposer.transform(
                DataGlobal.dataset.loc[self.data_ix].values),
                index=DataGlobal.dataset.loc[self.data_ix].index)
            #selcut = self._features_removal(int(cut_opt))
        else:
            selcut = DataGlobal.dataset.loc[self.data_ix]

        if proj.shape[1] != 2:
            mapping = self.interface.dim_red(
                metric=self.metric_map,
                n_components=2,
                min_dist=0.05,
                spread=1,
                n_neighbors=n_nei,
                n_epochs=self.epochs,
                learning_rate=self.lr,
                target_weight=self.super_w,
                verbose=False)

            if self.transform is not None:
                scft = selcut[~selcut.index.isin(self.transform)]
                sct = selcut[selcut.index.isin(self.transform)]

                mapping.fit(
                    scft,
                    y=functions.loc_cat(
                        DataGlobal.labels,
                        scft.index,
                        self.supervised))

                pj1 = self.interface.df.DataFrame(
                    mapping.transform(scft))
                # cudf workaround
                pj1.index = scft.index

                pj2 = self.interface.df.DataFrame(
                    mapping.transform(sct))
                # cudf workaround
                pj2.index = sct.index

                proj = self.interface.df.concat([pj1, pj2], axis=0)
                proj = proj.loc[selcut.index]
            else:

                proj = self.interface.df.DataFrame(
                    mapping.fit_transform(
                        selcut,
                        y=functions.loc_cat(
                            DataGlobal.labels,
                            selcut.index,
                            self.supervised)),
                    index=selcut.index)
                # for some reason it seems cudf doesn't assign the index
                # here...
                proj.index = selcut.index

            if self.save_map:
                with open(os.path.join(self.out_path,
                        'rc_data/' + self._name + '_2d.pkl'), 'wb') as file:
                    # keepfeat and decompt already in the not 2d map
                    pickle.dump(mapping, file)
                    file.close()

                proj.to_hdf(
                    os.path.join(self.out_path,
                        'rc_data/' + self._name + '_2d.h5'),
                    key='proj')

            del mapping

        """ Plot 2-dimensional umap of the optimal clusters. """

        plotting.plot_map(
            self.interface.get_value(
                proj,
                pandas=True),
            self.interface.get_value(
                clus_opt,
                pandas=True),
            'proj_clusters_' +
            self._name,
            self.out_path)

        """ Plot the same 2-dimensional umap with labels if provided. """

        if DataGlobal.labels is not None:
            
            plotting.plot_map(self.interface.get_value(proj, pandas=True), self.interface.get_value(
                DataGlobal.labels.loc[proj.index], pandas=True),
                'proj_labels_' + self._name, self.out_path)

        """ Plot the same 2-dimensional umap with transform only data if provided. """

        if self.transform is not None:
           
            transflab = self.interface.df.Series(
                ['fit-transform']*proj.shape[0], index=proj.index)
            transflab.loc[self.transform] = 'transform'

            plotting.plot_map(
                self.interface.get_value(proj,pandas=True),
                self.interface.get_value(transflab,pandas=True),
                'proj_trans_' + self._name,
                self.out_path)

    def _elbow(self, pj):
        """ Estimates the point of flex of a pairwise distances plot.

        Args:
            pj (pandas dataframe/numpy matrix): projection of saxmples in the low-dimensionality space
                obtained with UMAP, or adjacency matrix if SNN.

        Returns:
            (float): elbow value.
        """

        if self.clu_algo=='SNN':
            
            mat=self.interface.num.copy(pj)
            self.interface.num.fill_diagonal(mat,self.interface.num.inf)
            neigh=self.interface.num.nanmin(mat,axis=1)
            del mat

            #alternative to copying the matrix and filling the diagonal
            #n=pj.shape[0]
            #neigh=self.interface.num.min(self.interface.num.lib.stride_tricks.as_strided(pj, (n-1,n+1), (pj.itemsize*(n+1), pj.itemsize))[:,1:], axis=0)
            if not isinstance(neigh, self.interface.df.DataFrame):
                neigh = self.interface.df.DataFrame(neigh, columns=['elbow'])

        else:
            neigh = self.interface.n_neighbor(
                n_neighbors=self.min_sam_dbscan,
                metric=self.metric_clu,
                metric_params=self.mparams,
                n_jobs=-1).fit(pj)
            neigh = neigh.kneighbors(pj, return_distance=True)[0]
            
            if not isinstance(neigh, self.interface.df.DataFrame):
                neigh = self.interface.df.DataFrame(neigh)
            neigh['elbow']=neigh.iloc[:,1:].mean(axis=1)
            #neigh.columns = ['0', 'elbow']
        
        neigh = neigh.sort_values('elbow')
        neigh['delta'] = neigh['elbow'].diff().shift(periods=-1) \
            + neigh['elbow'].diff().shift(periods=+1) \
            - 2 * neigh['elbow'].diff()

        # return neigh['elbow'].iloc[neigh['delta'].idxmax()]
        # cuDF doesn't have idxmax, so here is a probably quite expensive
        # workaround
        neigh = neigh.sort_values('delta').dropna()
       
        #an issue with elbow position being at zero arises 
        #occasionally with SNN
        #this shouldn't happen unless points are overlapping
        #fix itm but for now use this workaround
        
        pos=0
        i=0
        while pos==0 and i>-neigh.shape[0]:
            pos=neigh['elbow'].iloc[i-1]
            i=-1

        if pos==0:
            #random value, if you get here probably something is wrong
            #with your data
            return 0.01
        else:
            return pos

    def _guess_parm(self, pj):
        """ Estimate a range for the clustering identification parameter.

        Args:
            pj (pandas dataframe): projection of saxmples in the low-dimensionality space
                obtained with UMAP.

        Returns:
            (numpy range): estimated range.
        """

        """ Use pairwise knn distances elbow method for DBSCAN;
            Take the square root of the total population for HDBSCAN."""

        if self.clu_algo == 'louvain':
            logging.log(DEBUG_R, 
                'Resolution range guess: [{:.5f},{:.5f}]'.format(
                    0, 5))
            return self.interface.num.linspace(0, 5, 6)

        if self.clu_algo  in ['DBSCAN', 'SNN']:
            ref = self._elbow(pj)
            logging.log(DEBUG_R, 
                'Epsilon range guess: [{:.5f},{:.5f}]'.format(
                    ref / 50, ref * 1.5))

            return self.interface.num.linspace(ref / 50, ref * 1.5, 100)

        elif self.clu_algo == 'HDBSCAN':
            minbound = self.interface.num.amax(
                [self.min_sam_dbscan, int(self.interface.num.sqrt(pj.shape[0] / 25))])
            maxbound = self.interface.num.amin(
                [250, int(self.interface.num.sqrt(pj.shape[0] * 2.5))])
            if minbound == maxbound:
                maxbound = minbound + 2

            step = int((maxbound - minbound) / 50)
            if step < 1:
                step = 1
            logging.log(DEBUG_R, 
                'Minimum samples range guess: [{:d},{:d}] with a {:d} point(s) step'.format(
                    self.interface.get_value(minbound),
                    self.interface.get_value(maxbound),
                    self.interface.get_value(step)))

            return self.interface.num.linspace(
                minbound, maxbound, (maxbound - minbound) / step, endpoint=False)

        else:
            sys.exit('ERROR: clustering algorithm not recognized')

    def snn(self, points, num_neigh):
        """ Calculates Shared Nearest Neighbor (SNN) matrix

        Args:
            points (dataframe or matrix): points coordinates.
            num_neigh (int): number of neighbors considered
                to define the similarity of two points.

        Returns:
            (matrix): SNN matrix as input for DBSCAN.
        """

        neigh = self.interface.n_neighbor(
            n_neighbors=num_neigh+1,
            metric=self.metric_clu,
            metric_params=self.mparams,
            n_jobs=-1).fit(points)
       
        allnei=neigh.kneighbors(points, return_distance=False)
        if self.gpu:
            allnei=allnei.drop(0,axis=1)
            neighlist = [self.interface.set(allnei.loc[x].values) for x in self.interface.get_value(allnei.index)]
        else:
            neighlist = [self.interface.set(x[1:]) for x in allnei]
        return 1-self.interface.num.asarray([[len(i.intersection(j))/num_neigh for j in neighlist] for i in neighlist])

    def _find_clusters(self, pj, cparm, cse=None, algorithm=None):
        """ Runs the selected density-based clusters identification algorithm.

        Args:
            pj (dataframe or matrics): points coordinates.
            cparm (float): clustering parameter.
            cse (int): value of clustering_selection_epsilon for HDBSCAN.
            algorithm (string): value of algorithm for HDBSCAN.

        Returns:
            (list of int): list of assigned clusters. 
        """


        if self.clu_algo in ['DBSCAN','SNN']:
            return self.interface.cluster(pj,
                eps=cparm,
                min_samples=self.min_sam_dbscan,
                metric='precomputed',
                n_jobs=-1,
                leaf_size=15)
        
        if self.clu_algo == 'louvain':
            return self.interface.cluster_louvain(pj,
                resolution=cparm)
            
        if self.clu_algo == 'HDBSCAN':
            clu_algo = HDBSCAN(
                algorithm=algorithm,
                alpha=1.0,
                approx_min_span_tree=True,
                gen_min_span_tree=False,
                leaf_size=15,
                allow_single_cluster=False,
                metric=self.metric_clu,
                metric_params=self.mparams,
                min_cluster_size=self.min_sam_dbscan,
                min_samples=int(cparm),
                cluster_selection_epsilon=cse,
                p=None).fit(pj)
            return clu_algo.labels_
        
        sys.exit('ERROR: clustering algorithm not recognized')

    def _run_single_instance(self, cutoff, nn):
        """ Run a single instance of clusters search for a given features cutoff and
        UMAP nearest neighbors number.

        Args:
            cutoff (float): features cutoff.
            nn (int): UMAP nearest neighbors value.

        Returns:
            (tuple (float, pd.Series, float, pd.DataFrame, pd.Index, tsvd object)): a tuple containing
            the silhoutte score corresponding to the best set of parameters;
            a series with the cluster membership identified for each sample;
            the optimal clustering parameter value found;
            a low dimensionality data projection from UMAP;
            a set of genes kept after low /MAD removal, nan if 'tSVD';
            the trained tsvd instance, None if 'variance'/'MAD'.
        """
            
        sil_opt = self.baseline

        labs_opt = [0] * DataGlobal.dataset.loc[self.data_ix].shape[0]
        cparm_opt = self.interface.num.nan
        keepfeat = self.interface.num.nan

        init = 'spectral'
        if DataGlobal.dataset.loc[self.data_ix].shape[0] <= self.dim + 1:
            init = 'random'
        logging.log(DEBUG_R, 'Initialization: ' + init)

        """ Remove columns with low information from data matrix. """

        logging.log(DEBUG_R, 'Features cutoff: {:.3f}'.format(cutoff))

        data_cut, decomposer = self._features_removal(cutoff)

        if self.filter_feat in ['variance', 'MAD']:
            keepfeat = data_cut.columns

        if self.norm is not None:

            """ Normalize data. """

            logging.log(DEBUG_R, 'Normalize with ' + self.norm)

            data_cut = self.interface.df.DataFrame(
                self.interface.norm(
                    data_cut,
                    norm=self.norm),
                index=data_cut.index,
                columns=data_cut.columns)

        """ Project data with UMAP. """

        logging.log(DEBUG_R, 'Number of nearest neighbors: {:d}'.format(nn))


        if (self.dim == data_cut.shape[1] and self.skip_equal_dim) or\
            self.skip_dimred:

            """ If the embedding space dimensionality corresponds to the data
                dimensionality, do not run the projection. """

            logging.info('Skipping non-linear dimensionality reduction step')

            mapping = classes.IdentityProjection()

        else:
            mapping = self.interface.dim_red(
                metric=self.metric_map,
                n_components=self.dim, min_dist=0.0, spread=1,
                n_neighbors=nn, n_epochs=self.epochs,
                learning_rate=self.lr, verbose=False,
                target_weight=self.super_w,
                random_state=self._umap_rs, init=init)

        if self.transform is not None:

            untransf = data_cut[~data_cut.index.isin(self.transform)]

            pj = self.interface.df.DataFrame(
                mapping.fit_transform(
                    untransf,
                    y=functions.loc_cat(
                        DataGlobal.labels,
                        untransf.index,
                        self.supervised)))
            # cudf workaround
            pj.index = untransf.index

        else:

            pj = self.interface.df.DataFrame(
                mapping.fit_transform(
                    data_cut,
                    y=functions.loc_cat(
                        DataGlobal.labels,
                        data_cut.index,
                        self.supervised)))
            # cudf workaround
            pj.index = data_cut.index

        if not pj.isnull().values.any():

            """ Set cluster_selection_epsilon for HDBSCAN. """

            cse = None
            hdbalgo = 'best'
            if self.clu_algo == 'HDBSCAN':
                cse = float(self._elbow(pj))
                if self.metric_clu == 'cosine':
                    hdbalgo = 'generic'
                    pj = pj.astype(self.interface.num.float64)
           
            self.mparams = {}
            if self.metric_clu == 'mahalanobis':
                try:
                    self.mparams = {
                        'VI': self.interface.num.linalg.inv(
                            self.interface.num.cov(pj.T))}
                except BaseException:
                    self.mparams = {
                        'VI': self.interface.num.linalg.pinv(
                            self.interface.num.cov(pj.T))} 
 
            if self.clu_algo == 'DBSCAN':
                to_cluster=self.interface.pwd(pj,
                    metric = self.metric_clu,
                    **self.mparams,
                    n_jobs=-1)
            elif self.clu_algo == 'SNN':
                to_cluster=self.snn(pj,nn)
            elif self.clu_algo == 'louvain':
                to_cluster=1-self.snn(pj,nn)
                if self.gpu:
                    to_cluster=self.interface.build_graph(to_cluster)
            else:
                to_cluster=pj

            """ Set clustering parameter range at the first iteration. """
            
            # TODO: check if better to update at every iteration
            cparm_range = self.cparm_range
            if cparm_range == 'guess':
                cparm_range = self._guess_parm(to_cluster)

            # Note: Calculating (H)DBSCAN on a grid of parameters is cheap even
            # with Differential Evolution.

            for cparm in cparm_range:  # self.cparm_range

                logging.log(DEBUG_R, 
                    'Clustering parameter: {:.5f}'.format(
                        self.interface.get_value(cparm)))

                labs = self._find_clusters(to_cluster, cparm, cse, hdbalgo)

                # not 100% sure about this, keep until weights on noise will be
                # added
                compset = self.interface.set(labs)
                compset.discard(-1)
                
                #if too many were discarded take another
                labs  = self.interface.df.Series(labs, index=pj.index)
                ratio = labs.value_counts()
                ratio = ratio[-1]/labs.shape[0] if -1 in ratio else 0
                
                #could be stricter/looser
                if len(compset) > 1 and ratio<=.3:
                    sil = functions.calc_score(pj, labs,
                        self.score, self.metric_clu, self.interface)
                else:
                    if ratio<=.3:
                        logging.log(DEBUG_R,
                            'Too many points discarded ({:d}%>{:d}%)!'.format(int((1-ratio)*100),70))
                    sil = self.baseline

                if sil > sil_opt:
                    cparm_opt = cparm
                    sil_opt = sil
                    labs_opt = labs

        return sil_opt, labs_opt,\
            cparm_opt, pj, mapping, keepfeat, decomposer

    def _objective_function(self, params):
        """ Objective function for Differential Evolution.

        Args:
            params (list): a list containing a single feature cutoff and a
                UMAP nearest neighbors parameter.

        Returns:
            (tuple (float, pd.Series, float, pd.DataFrame, pd.Index, tsvd object)): a tuple containing
            the loss value for the given set of parameters;
            a series with the cluster membership identified for each sample;
            the optimal clustering parameter value found;
            a low dimensionality data projection from UMAP;
            a set of genes kept after low variance/MAD removal, nan if tSVD;
            the trained tsvd instance, None if 'variance'/'MAD'.
        """

        sil_opt, labs, cparm_opt, pj, mapping, keepfeat,\
            decomposer = self._run_single_instance(
            params[0], int(params[1]))

        return 1 - sil_opt, labs, cparm_opt, pj, mapping, keepfeat, decomposer

    def _run_grid_instances(self, nnrange):
        """ Run Grid Search to find the optimal set of parameters by maximizing the
        clustering score.

        Args:
            nnrange (numpy range): UMAP nearest neighbors range.

        Returns:
            (tuple (list of floats, list of objects, list of floats)): a tuple containing
            the list of best parameters;
            a list containing score, labels, clustering parameter,
            projected points, trained maps, filtered features and
            trained low-information filter from the best scoring model;
            a matrix containing all the explored models' parameters
            and their scores (useful for plotting the hyperspace).
        """

        # Note: this should be moved to optimizers.

        sil_opt = self.baseline
        keepfeat = []
        decomp_opt = None
        labs_opt = [0] * DataGlobal.dataset.loc[self.data_ix].shape[0]
        if self.transform is not None:
            labs_opt = labs_opt[:len(labs_opt)-len(self.transform)]
        cparm_opt = self.interface.num.nan
        nei_opt = DataGlobal.dataset.loc[self.data_ix].shape[0]
        
        if self.filter_feat in ['variance', 'MAD']:
            cut_opt = 1.0
        elif self.filter_feat == 'tSVD':
            cut_opt = DataGlobal.dataset.loc[self.data_ix].shape[1]
        scoreslist = [[], [], []]

        init = 'spectral'
        if DataGlobal.dataset.loc[self.data_ix].shape[0] <= self.dim + 1:
            init = 'random'
        logging.log(DEBUG_R, 'Initialization: ' + init)

        if self.ffrange == 'kde':
            self.ffrange = ['kde']

        for cutoff in self.ffrange:

            """ Remove columns with low information from data matrix. """

            logging.log(DEBUG_R, 'Features cutoff: {:.3f}'.format(cutoff))

            data_cut, decomposer = self._features_removal(cutoff)

            if self.norm is not None:

                """ Normalize data. """

                logging.log(DEBUG_R, 'Normalize with ' + self.norm)

                data_cut = self.interface.df.DataFrame(
                    self.interface.norm(
                        data_cut,
                        norm=self.norm),
                    index=data_cut.index,
                    columns=data_cut.columns)

            for nn in nnrange:

                """ Project data with UMAP. """

                logging.log(DEBUG_R, 'Number of nearest neighbors: {:d}'.format(nn))

                if (self.dim == data_cut.shape[1] and self.skip_equal_dim) or\
                    self.skip_dimred:

                    """ If the embedding space dimensionality corresponds to the data
                        dimensionality, do not run the projection. """

                    logging.info('Skipping non-linear dimensionality reduction step')

                    mapping = classes.IdentityProjection()

                else:
                    mapping = self.interface.dim_red(
                        metric=self.metric_map,
                        n_components=self.dim,
                        min_dist=0.0, spread=1, n_neighbors=nn,
                        n_epochs=self.epochs, learning_rate=self.lr,
                        target_weight=self.super_w,
                        verbose=False, random_state=self._umap_rs,
                        init=init)

                """if data to be projected only is provided, calculate optimality
                   only on the fit data. """

                if self.transform is not None:

                    untransf = data_cut[~data_cut.index.isin(self.transform)]

                    pj = self.interface.df.DataFrame(
                        mapping.fit_transform(
                            untransf,
                            y=functions.loc_cat(
                                DataGlobal.labels,
                                untransf.index,
                                self.supervised)))
                    # cudf workaround
                    pj.index = untransf.index

                else:
                    
                    pj = self.interface.df.DataFrame(
                        mapping.fit_transform(
                            data_cut,
                            y=functions.loc_cat(
                                DataGlobal.labels,
                                data_cut.index,
                                self.supervised)))
                    # cudf workaround
                    pj.index = data_cut.index

                if not pj.isnull().values.any():

                    """ Set cluster_selection_epsilon for HDBSCAN. """

                    cse = None
                    hdbalgo = 'best'
                    if self.clu_algo == 'HDBSCAN':
                        cse = float(self._elbow(pj))
                        if self.metric_clu == 'cosine':
                            hdbalgo = 'generic'
                            pj = pj.astype(self.interface.num.float64)

                    scoreslist[0].append(cutoff)
                    scoreslist[1].append(nn)
                    scoreslist[2].append(self.baseline)

                    self.mparams = {}
                    if self.metric_clu == 'mahalanobis':
                        try:
                            self.mparams = {
                                'VI': self.interface.num.linalg.inv(
                                    self.interface.num.cov(pj.T))}
                        except BaseException:
                            self.mparams = {
                                'VI': self.interface.num.linalg.pinv(
                                    self.interface.num.cov(pj.T))}

                    """ Build the distance matrix for DBSCAN, or the 
                        adjacency matrix for SNN and louvain """

                    if self.clu_algo == 'DBSCAN':
                        to_cluster=self.interface.pwd(pj,
                            metric = self.metric_clu,
                            **self.mparams,
                            n_jobs=-1)
                    elif self.clu_algo == 'SNN':
                        to_cluster=self.snn(pj,nn)
                    elif self.clu_algo == 'louvain':
                        to_cluster=1-self.snn(pj,nn)
                        if self.gpu:
                            to_cluster=self.interface.build_graph(to_cluster)
                    else:
                        to_cluster=pj

                    """ Set clustering parameter range at the first iteration. """

                    cparm_range = self.cparm_range
                    if cparm_range == 'guess':
                        cparm_range = self._guess_parm(to_cluster)
                    # Note: Calculating (H)DBSCAN on a grid of parameters is
                    # cheap even with Differential Evolution.

                    for cparm in cparm_range:  # self.cparm_range

                        logging.log(DEBUG_R, 
                            'Clustering parameter: {:.5f}'.format(
                                self.interface.get_value(cparm)))
                        
                        labs = self._find_clusters(to_cluster, cparm, cse, hdbalgo)
                        
                        # not 100% sure about this, keep until weights on noise
                        # will be added
                        compset = self.interface.set(labs)
                        compset.discard(-1)

                        #if too many were discarded take another
                        labs  = self.interface.df.Series(labs, index=pj.index)
                        ratio = labs.value_counts()
                        ratio = ratio[-1]/labs.shape[0] if -1 in ratio else 0
                        
                        #could be stricter/looser
                        if len(compset) > 1 and ratio<=self.noise_ratio:
                            sil = functions.calc_score(pj, labs,
                                self.score, self.metric_clu, self.interface)
                        else:
                            if ratio<=self.noise_ratio:
                                logging.log(DEBUG_R,
                                    'Too many points discarded ({:d}%>{:d}%)!'.format(int((1-ratio)*100),70))
                            sil = self.baseline


                        logging.log(DEBUG_R, 'Clustering score: {:.3f}'.format(sil))

                        if sil > scoreslist[2][-1]:
                            scoreslist[2].pop()
                            scoreslist[2].append(sil)

                        if sil > sil_opt:
                            cparm_opt = cparm
                            sil_opt = sil
                            pj_opt = pj
                            labs_opt = labs
                            nei_opt = nn
                            cut_opt = cutoff
                            map_opt = mapping
                            if self.filter_feat in [
                                    'variance', 'MAD', 'correlation']:
                                keepfeat = data_cut.columns
                            if self.filter_feat == 'tSVD':
                                decomp_opt = decomposer

        """ If an optimal solution was not found. """

        if sil_opt == self.baseline:
            logging.info('Optimal solution not found!')
            pj_opt = pj
            map_opt = mapping

        return [cut_opt, nei_opt] , [sil_opt, labs_opt,\
               cparm_opt, pj_opt, map_opt, keepfeat, decomp_opt],\
               scoreslist

    def _optimize_params(self):
        """ Wrapper function for the parameters optimization.

        Returns:
            (tuple (float, pd.Series, float, int, int, pd.DataFrame, 
                float, pd.Index, tsvd onject, float, list of floats)): a tuple containing
                the silhoutte score corresponding to the best set of parameters;
                a series with the cluster membership identified for each sample;
                the optimal clusters identification parameter value found;
                the  total number of clusters determined by the search;
                the optimal number of nearest neighbors used with UMAP;
                a low dimensionality data projection from UMAP;
                the optimal cutoff value used for the features removal step;
                the set of genes kept after low variance/MAD removal, nan if tSVD;
                the trained tsvd instance, None if 'variance'/'MAD';
                the percentage of points forecefully assigned to a class if outliers='reassign';
                the list of all scores evaluated and their parameters.
        """

        logging.info(
            'Dimensionality of the target space: {:d}'.format(
                self.dim))
        logging.info('Samples #: {:d}'.format(
            DataGlobal.dataset.loc[self.data_ix].shape[0]))
        if self.dyn_mesh:
            if self.optimizer == 'grid':
                logging.info('Dynamic mesh active, number of grid points: {:d}'.format(
                        self.nei_points[0] * self.ffpoints[0]))
            if self.optimizer == 'de':
                logging.info('Dynamic mesh active, number of candidates: {:d} \
                              and iterations: {:d}'.format(
                        self.search_candid[0], self.search_iter[0]))

        if self.transform is not None:
            logging.info('Transform-only Samples #: {:d}'.format(
                    len(self.transform)))
       
        numpoints = DataGlobal.dataset.loc[self.data_ix].shape[0]
        
        if self.transform is not None:
            numpoints -= len(self.transform)

        if numpoints != 0:
            
            if self.nei_range == 'logspace':
                if self.nei_factor >= 1:
                    minbound = self.interface.num.log10(
                        self.interface.num.sqrt(numpoints - 1))
                    maxbound = self.interface.num.log10(numpoints - 1)
                else:
                    minbound = self.interface.num.log10(
                        self.interface.num.sqrt(numpoints * self.nei_factor - 1))
                    maxbound = self.interface.num.log10(numpoints * self.nei_factor - 1)
                
                """ Neighbors cap. """

                if self.neicap is not None:
                    if minbound > self.interface.num.log10(self.neicap):
                        minbound = self.interface.num.log10(self.neicap / 10)
                    if maxbound > self.interface.num.log10(self.neicap):
                        maxbound = self.interface.num.log10(self.neicap)

                """ Hard limit. """

                if minbound <= 0:
                    minbound = self.interface.num.log10(2)

                nnrange = sorted([x for x in self.interface.num.logspace(
                    minbound, maxbound, num=self.nei_points[0])])
            
            elif hasattr(self.nei_range, '__call__'):
                """ If a function to select the neighbors was selected. """

                nnrange = self.nei_range(numpoints * self.nei_factor)
            
            else:
                nnrange = self.nei_range
            
            if not isinstance(nnrange, list):
                nnrange = [nnrange]

            #check 
            logging.debug('neihgbours range:')
            logging.debug(nnrange)

            """ Number of neighbors cannot be more than provided cap. """
            """ Number of neighbors cannot be more than the number of samples. """
            """ Number of neighbors cannot be less than 2. """
            """ Make sure they are all integers. """
            
            maxnei = self.neicap if self.neicap is not None and self.neicap < numpoints-1 else numpoints-1

            nnrange = sorted(list(set(
                [int(max(2,min(x,maxnei))) for x in nnrange])))
            

            #if self.neicap is not None:
            #    nnrange = sorted(list(self.interface.set(
            #        [x if x <= self.neicap else self.neicap for x in nnrange])))
            
            #""" Number of neighbors cannot be more than the number of samples. """
            
            #nnrange = sorted(list(self.interface.set(
            #    [x if x < numpoints else numpoints-1 for x in nnrange])))

            #""" Number of neighbors cannot be less than 2. """

            #nnrange = sorted(list(self.interface.set(
            #    [x if x > 2 else 2 for x in nnrange])))

            #""" Make sure they are all integers. """

            #nnrange = sorted(list(self.interface.set(
            #    [int(x) for x in nnrange])))

        else:

            nnrange = []
        
        """ Run Optimizer. """

        if self.optimizer == 'grid':

            """ Grid Search. """

            logging.info('Running Grid Search...')

            config_opt, results_opt, scoreslist = self._run_grid_instances(
                nnrange)

            logging.info('Done!')

        elif self.optimizer == 'de':

            """ Differential Evolution. """

            logging.info('Running Differential Evolution...')

            # Note: this works as monodimensional DE, but may be slightly
            # inefficient
            bounds = [(min(self.ffrange), max(self.ffrange)),
                      (min(nnrange), max(nnrange))]
            config_opt, results_opt, scoreslist = de._differential_evolution(
                self._objective_function, bounds, maxiter=self.search_iter[0],
                n_candidates=self.search_candid[0], integers=[False, True], seed=self._seed)

            logging.info('Done!')

        elif self.optimizer == 'tpe':

            """ Tree-structured Parzen Estimators with Optuna. """

            logging.info('Running TPE with Optuna...')
 
            bounds = [(min(self.ffrange), max(self.ffrange)),
                      (min(nnrange), max(nnrange))]

            config_opt, results_opt, scoreslist = tpe._optuna_tpe(
                self._objective_function, bounds,
                n_candidates=self.search_candid[0], 
                patience=self.tpe_patience, seed=self._seed)

            logging.info('Done!')
            
        else:
            sys.exit('ERROR: optimizer not recognized')

        """ Split the output """

        cut_opt, nei_opt = config_opt
        sil_opt, labs_opt, cparm_opt, pj_opt, map_opt,\
            keepfeat, decomp_opt = results_opt

        if not isinstance(labs_opt,self.interface.df.Series):
            labs_opt=self.interface.df.Series(labs_opt, index=pj_opt.index)

        """ If data to be projected only is provided, apply projection. """

        if self.transform is not None:

            #if no clustering was found
            if len(keepfeat) == 0:
               labs_new = self.interface.df.Series([0]*len(self.transform),
                    index=self.transform)
               #labs_opt=self.interface.df.concat([labs_opt,labs_new], axis=0)

            else:
                # A bit redundant, try to clean up
                if self.filter_feat in ['variance', 'MAD', 'correlation']:
                    transdata = DataGlobal.dataset[keepfeat].loc[self.transform]
                elif self.filter_feat == 'tSVD':
                    transdata = decomp_opt.transform(
                        DataGlobal.dataset.loc[self.transform])
                
                #pj_t=self.interface.df.DataFrame(map_opt.transform(transdata),
                #    index=self.transform)
                #cudf workaround
                pj_t = self.interface.df.DataFrame(map_opt.transform(transdata))
                pj_t.index = self.transform

                pj_opt = self.interface.df.concat([pj_opt, pj_t],
                    axis=0)

                logging.log(DEBUG_R, 
                    'Transform-only data found at this level: membership will be assigned with KNN')

                """ Assign cluster membership with k-nearest neighbors. """
                 
                labs_new = local_KNN(pj_opt, 
                    functions.one_hot_encode(labs_opt, self._name,
                    self.interface, min_pop=self.min_csize),
                    nei_opt, self.metric_clu, 
                    self.interface, as_series=True)

                #labs_new = self.interface.df.Series(
                #    local_KNN(pj_opt, 
                #    functions.one_hot_encode(labs_opt, self.min_sam_dbscan, self._name,
                #    self.interface),
                #    nei_opt, self.metric_clu, 
                #    self.interface, as_series=True),
                #    index=self.transform)

            labs_opt = self.interface.df.concat([labs_opt,labs_new], axis=0)

        """ Dealing with discarded points if outliers!='ignore'
            applies only if there's more than one cluster identified
            and if at least 10% but less than 90% of the samples have been discarded.
        """

        reassigned = 0.0
        if self.outliers == 'reassign' \
            and -1 in labs_opt.values \
            and labs_opt.unique().shape[0] > 2:

            if labs_opt.value_counts().to_dict()[-1] > labs_opt.shape[0] * .1 \
            and labs_opt.value_counts().to_dict()[-1] < labs_opt.shape[0] * .9:

                labs_out = labs_opt[labs_opt != -1]
                reassigned = (
                    labs_opt.shape[0] - labs_out.shape[0]) * 1.0 / labs_opt.shape[0]
                #labs_opt = self._KNN(nei_opt, pj_opt, labs_out,
                #                    cutoff=.5).loc[labs_opt.index]
                labs_new = local_KNN(pj_opt, 
                    functions.one_hot_encode(labs_out, self._name,
                    self.interface, min_pop=self.min_csize),
                    nei_opt, self.metric_clu, 
                    self.interface, as_series=True) 
                                    
                labs_opt = self.interface.df.concat([labs_out,labs_new], axis=0)

        num_clus_opt = len(self.interface.set(labs_opt)) - \
            (1 if -1 in labs_opt else 0)

        logging.info(
            '\n=========== Optimization Results ' +
            self._name +
            ' ===========\n' +
            'Features # Cutoff: {:.5f}'.format(cut_opt) + '\n' +
            'Nearest neighbors #: {:d}'.format(nei_opt) + '\n' +
            'Clusters identification parameter: {:.5f}'.format(
                self.interface.get_value(cparm_opt)) + '\n' +
            'Clusters #: {:d}'.format(num_clus_opt) + '\n')

        return sil_opt, labs_opt, cparm_opt, num_clus_opt, nei_opt, pj_opt,\
               cut_opt, map_opt, keepfeat, decomp_opt, reassigned, scoreslist

    def iterate(self):
        """ Iteratively  clusters the input data, by first optimizing the parameters,
        binarizing the resulting labels, plotting and repeating. 
        """

        #self._depth += 1

        if self._level_check():
            self.clust_opt=None
            return

        if DataGlobal.dataset.loc[self.data_ix].shape[0] < self.pop_cut:
            logging.info('Population too small!')
            self.clus_opt = None
            return 

        minimum, clus_tmp, chosen, n_clu, n_nei, pj, cut, chomap, keepfeat,\
        decomp_opt, reassigned, scoreslist = self._optimize_params()

        """ Save cluster best parameters to table. """

        vals = [self._name,
                DataGlobal.dataset.loc[self.data_ix].shape[0],
                n_clu, self.dim, minimum, n_nei,
                chosen, cut, self.metric_map,
                self.metric_clu, self.norm, reassigned, self._seed]

        with open(os.path.join(self.out_path, 'rc_data/paramdata.csv'), 'a') as file:
            writer = csv.writer(file)
            writer.writerow(vals)
            file.close()

        """ Save intermediate UMAP data to be able to re-access specific clusters
        (expensive, only if debug True). 
        """

        if self.save_map:
            with open(os.path.join(self.out_path,
                    'rc_data/' + self._name + '.pkl'), 'wb') as file:
                if self.filter_feat in ['variance', 'MAD', 'correlation']:
                    pickle.dump([keepfeat, chomap], file)
                elif self.filter_feat == 'tSVD':
                    pickle.dump([decomp_opt, chomap], file)
                file.close()
            pj.to_hdf(
                os.path.join(self.out_path,
                    'rc_data/' + self._name + '.h5'),
                key='proj')

        del chomap

        if n_clu < 2:
            logging.info('Optimal solution has only one cluster!')
            self.clus_opt = None
            return

        """ Plotting. """

        self._plot(n_nei, pj, cut, keepfeat, decomp_opt, clus_tmp, scoreslist)

        """ Binarize data. """

        #TEST REMOVE IF CREATES ISSUES
        
        clus_tmp = clus_tmp.astype(int)
        clus_tmp = functions.one_hot_encode(clus_tmp, self._name,
                    self.interface, min_pop=self.min_csize)

        """ Checkpoint. """

        if self.chk:
            
            if clus_tmp is not None:
                if OptionalImports.feather:
                    clus_tmp.reset_index().to_feather(
                        os.path.join(
                            self.out_path, 'rc_data/chk/clusters_chk_'+self._name+'.fe'))
                else:
                    clus_tmp.to_hdf(
                        os.path.join(
                            self.out_path, 'rc_data/chk/clusters_chk_'+self._name+'.h5'),
                            key='df')
            
        """ Dig within each subcluster and repeat. """

        for l in list(clus_tmp.columns):

            sel_new = DataGlobal.dataset.loc[self.data_ix].loc[clus_tmp[clus_tmp[l] == 1].index]


            logging.info('Going deeper within Cluster # ' +
                         str(l) + ' [depth: {:d}'.format(self._depth+1) + ']')

            to_transform = None
            if self.transform is not None:
                to_transform = [x for x in self.transform if x in sel_new.index]
                if len(to_transform) == 0:
                    to_transform = None
           
            #if you got a class of just transforms (weird! shouldn't happen!) check this!
            if to_transform is not None and len(to_transform)==len(sel_new.index):
                logging.warning('Found a class with transform-only data, ' +
                                'something must have gone wrong along the way.')
                continue

            """ Move along the list of parameters to change granularity. """
            if self.optimizer == 'grid' and len(self.ffpoints) > 1:
                self.ffpoints = self.ffpoints[1:]
                logging.info(
                    'Parameters granilarity change ' +
                    '[features filter: {:d}'.format(
                        self.interface.get_value(self.ffpoints[0])) + ']')
            if self.optimizer == 'grid' and len(self.nei_points) > 1:
                self.nei_points = self.nei_points[1:]
                logging.info(
                    'Parameters granilarity change ' +
                    '[nearest neighbors: {:d}'.format(
                        self.interface.get_value(self.nei_points[0])) + ']')
            if self.optimizer in ['de','tpe'] and len(self.search_candid) > 1:
                self.search_candid = self.search_candid[1:]
                logging.info('Parameters granilarity change ' +
                             '[Candidates population: {:d}'.format(int(self.search_candid[0])) + ']')
            if self.optimizer == 'de' and len(self.search_iter) > 1:
                self.search_iter = self.search_iter[1:]
                logging.info('Parameters granilarity change ' +
                             '[DE iterations: {:d}'.format(int(self.search_iter[0])) + ']')

            deep = IterativeClustering(sel_new.index, lab=None, transform=to_transform,
                dim=self.dim, epochs=self.epochs, lr=self.lr, nei_range=self.nei_range,
                nei_points=self.nei_points, neicap=self.neicap, 
                skip_equal_dim=self.skip_equal_dim, skip_dimred=self.skip_dimred, metric_map=self.metric_map,
                metric_clu=self.metric_clu, pop_cut=self.pop_cut, filter_feat=self.filter_feat,
                ffrange=self.ffrange, ffpoints=self.ffpoints, optimizer=self.optimtrue,
                search_candid=self.search_candid, search_iter=self.search_iter, 
                tpe_patience=self.tpe_patience, score=self.score, baseline=self.baseline, norm=self.norm,
                dyn_mesh=self.dyn_mesh, max_mesh=self.max_mesh, min_mesh=self.min_mesh,
                clu_algo=self.clu_algo, cparm_range=self.cparm_range, min_sam_dbscan=self.min_sam_dbscan,
                outliers=self.outliers, noise_ratio=self.noise_ratio, min_csize=self.min_csize, 
                name=str(l), debug=self.debug, max_depth=self.max_depth, save_map=self.save_map, 
                RPD=self.RPD, out_path=self.out_path, depth=self._depth+1, 
                chk=self.chk, gpu=self.gpu, _user=False)

            deep.iterate()

            if deep.clus_opt is not None:

                # for now join not available in cudf
                #clus_tmp = self.interface.df.concat([clus_tmp, deep.clus_opt], axis=1, join='outer')
                
                clus_tmp = self.interface.df.concat(
                    [clus_tmp, deep.clus_opt], axis=1)

                clus_tmp = clus_tmp.fillna(0).astype(int)
                
        self.clus_opt = clus_tmp

if __name__ == "__main__":

    pass
