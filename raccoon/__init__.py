
"""
RACCOON (Recursive Algorithm for Coarse-to-fine Clustering OptimizatiON)
F. Comitani     @2018-2021
A. Maheshwari   @2019
"""

import os
import sys

import csv
import pickle

import logging
DEBUG_R = 15

import time
import psutil

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

import raccoon.utils.plots as plotting
import raccoon.utils.functions as functions
import raccoon.utils.trees as trees
import raccoon.utils.de as de
import raccoon.utils.classification
from raccoon.utils.option import OptionalImports
import raccoon.interface as interface

""" Search for optional libraries.  """

try:
    import hdbscan as HDBSCAN
    OptionalImports.hdbscan = True
except ImportError:
    pass

__version__ = "0.3.0"

class DataGlobal:

    """ Static container for the input data to be filled
        by the user at the first iteration. """

    # Hopefully this saves a bit of memory

    dataset = None
    labels = None


class RecursiveClustering:

    """ To perform recursive clustering on a samples x features matrix. """

    def __init__(self, data, lab=None, transform=None, supervised=False, 
                supervised_weight=0.5, dim=2, epochs=5000,
                lr=0.05, neirange='logspace', neipoints=25, neifactor=1.0,
                neicap=250, metric_map='cosine', metric_clu='euclidean', popcut=50,
                filterfeat='variance', ffrange='logspace', ffpoints=25,
                optimizer='grid', depop=10, deiter=10, score='silhouette',
                norm=None, dynmesh=False, maxmesh=20, minmesh=4,
                clusterer='DBSCAN', cparmrange='guess', minclusize=10, outliers='ignore',
                fromfile=None,  # resume=False,
                name='0', debug=False, maxdepth=None, savemap=False, RPD=False,
                outpath="", depth=-1, gpu=False, _user=True):
        """ Initialize the the class.

        Args:
            data (matrix, pandas dataframe or pandas index): if first call (_user==True), input data
                in pandas dataframe-compatible format (samples as row, features as columns),
                otherwise index of samples to carry downstream during the recursion calls.
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
            epochs (integer): number of UMAP epochs.
            lr (float): UMAP learning rate.
            neirange (array, list of integers or string): list of nearest neighbors values to be
                used in the search;
                if 'logspace' take an adaptive range based on the dataset size at each iteration
                with logarithmic spacing (reccomended),
                if 'sqrt' always take the sqare root of the number of samples.
                if 'quartsqrt' always take the sqare root of half the number of samples
                (ideal for extremely large datasets).
            neipoints (int or list of int): number of grid points for the neighbors search,
                if list, each value will be subsequently used at the next iteration until
                all values are exhausted,
                (works only with neirange='logspace' default 25).
            neifactor (float): scaling factor for 'logspace' and 'sqrt' selections in neirange
            neicap (int): maximum number of neighbours (reccomended with low-memory systems,
                default 250).
            metric_map (string): metric to be used in UMAP distance calculations (default cosine).
            metric_clu (string): metric to be used in clusters identification and clustering score
                calculations (default euclidean)
                Warning: 'cosine' does not work with HDBSCAN, normalize to 'l2' and use 'euclidean'
                instead.
            popcut (integer): minimum number of samples for a cluster to be considered valid
                (default 50).
            filterfeat (string): set the method to filter features in preprocessing;
                if 'variance' remove low variance genes
                if 'MAD' remove low median absolute deviation genes
                if 'correlation' remove correlated genes
                if 'tSVD' use truncated single value decomposition (LSA)
            ffrange (array, list or string): if filterfeat=='variance'/'MAD'/'correlation',
                percentage values for the low-variance/correlation removal cufoff search;
                if 'logspace' (default) take a range between .3 and .9 with logarithmic spacing
                (reccomended, will take the extremes if optimizer=='de');
                if 'kde' kernel density estimation will be used to find a single optimal
                low-variance cutoff (not compatible with optimizer=='de')
                if filterfeat=='tSVD', values for the number of output compontents search;
                if 'logspace' (default) take a range between number of features times .3 and
                .9 with logarithmic spacing
                (reccomended, will take the extremes if optimizer=='de')
            ffpoins (int or list of int): number of grid points for the feature removal cutoff search
                if list, each value will be subsequently used at the next iteration until
                all values are exhausted,
                (works only with ffrange='logspace', default 25).
            optimizer (string): choice of parameters optimizer, can be either 'grid' for grid search,
                'de' for differential evolution, or 'auto' for automatic
                (default is 'grid').
                Automatic will chose between grid search and DE depending on the number of
                search points (de if >25), works only if dynmesh is True.
            depop (int or list of int): size of the candidate solutions population in
                differential evolution
                if list, each value will be subsequently used at the next iteration until
                all values are exhausted
                (works only with optimizer='de', default 10).
            deiter (int or list of int): maximum number of iterations of differential evolution
                if list, each value will be subsequently used at the next iteration until
                all values are exhausted
                (works only with optimizer='de', default 10).
            score (string): objective function of the optimization (currently only 'dunn'
                and 'silhouette' are available, default 'silhouette').
            norm (string): normalization factor before dimensionality reduction (default None),
                not needed if metric_map is cosine
                if None, don't normalize.
            dynmesh (bool): if true, adapt the number of mesh points (candidates and iteration in DE)
                to the population, overrides neipoints, depop, deiter and ffpoints (default false).
            maxmesh (int): maximum number of points for the dynmesh option (hit at 10000 samples,
                default 20), this is a single dimension, the actuall mesh will contain n*n points.
            minmesh (int): minimum number of points for the dynmesh option (hit at 50 samples,
                default 4, must be >3 if optimizer='de'),
                this is a single dimension, the actuall mesh will contain n*n points.
            clusterer (string): selects which algorithm to use for clusters identification.
                Choose among 'DBSCAN' (default), 'SNN' (Shared Nearest Neighbours DBSCAN),  HDBSCAN,
                or 'louvain' (Louvain community detection with SNN).
            cparmrange (array, list) or string: clusters identification parameter range to
                be explored (default 'guess').
                When 'DBSCAN' this corresponds to epsilon (if 'guess' attempts to identify it
                by the elbow method);
                When 'HDBSCAN' this corresponds to the minimum number of samples required by
                the clusters (if 'guess' adapts it on the dataset population).
            minclusize (int): minimum number of samples in a cluster used in DBSCAN and HDBSCAN
                (default is 10).
            outliers (string): selects how to deal with outlier points in the clusters assignment
                if 'ignore' discard them
                if 'reassign' try to assign them to other clusters with knn if more than 10%
                of the total population was flagged.
            fromfile (string): path to parmdata.csv file to load, if active it will overwrite all
                other selections to follow the loaded parameters, unless resume is active.
            resume (bool): if True, resume the search from a previous run (works only if
                fromfile is provided).
            name (string): name of current clustering level (should be left as default, '0',
                unless continuing from a previous run).
            debug (boolean): specifies whether algorithm is run in debug mode (default is False).
            maxdepth (int): Specify the maximum number of recursion iterations, if None (default),
                keep going while possible.
                0 stops the algorithm immediately, 1 stops it after the first level.
            savemap (boolean): if active, saves the trained maps to disk (default is False).
                Needed to run the k-NN classifier.
            RPD (boolean): specifies whether to save RPD distributions for each cluster
                (default is False). Warning: this option is unstable and not reccomended.
            outpath (string): path to the location where outputs will be saved (default,
                save to the current folder).
            depth (integer): current depth of recursion (should be left as default,
                -1, unless continuing from a previous run).
            gpu (bool): Activate GPU version (requires RAPIDS).
            _user (bool): Boolean switch to distinguish initial user input versus recursion
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
                            data = self.interface.df.from_pandas(data)
                        except BaseException:
                            pass
                    print("Unexpected error: ", sys.exc_info()[0])
                    print("Input data should be in a format that can be translated \
                           to pandas/cuDF dataframe!")
                    raise

            DataGlobal.dataset = data
            data = data.index

            if lab is not None and not isinstance(
                    lab, self.interface.df.Series):
                try:
                    lab = self.interface.df.Series(lab)
                except BaseException:
                    print("Unexpected error: ", sys.exc_info()[0])
                    print("Labels data should be in a format that can be translated \
                          to pandas series!")
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
            warnings.warn("Labels need to be provided for supervised dimensionality \
                           reduction, setting supervised to False.")
            self.supervised = False

        self.super_w = supervised_weight
        self.optimizer = optimizer
        # Keep track of the original optimizer if changed
        self.optimtrue = optimizer

        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.neirange = neirange
        self.neipoints = neipoints
        self.neifactor = neifactor
        self.neicap = neicap
        self.metric_map = metric_map
        self.metric_clu = metric_clu
        self.mparams = {}
        self.popcut = popcut
        self.filterfeat = filterfeat
        self.ffrange = ffrange
        self.ffpoints = ffpoints
        self.debug = debug
        self.savemap = savemap
        self.maxdepth = maxdepth
        self.RPD = RPD
        self.depop = depop
        self.deiter = deiter
        self.outpath = outpath
        self.clus_opt = None
        self._name = name
        self._depth = depth
        self.score = score
        self.norm = norm
        self.dynmesh = dynmesh
        self.maxmesh = maxmesh
        self.minmesh = minmesh

        self.clusterer = clusterer
        self.cparmrange = cparmrange
        self.minclusize = minclusize
        self.outliers = outliers

        self.fromfile = fromfile
        #self.resume = resume

        """ CPU vs GPU methods check. """

        if self.gpu:
            if self.clusterer not in  ['DBSCAN', 'louvain']:
                warnings.warn("Only DBSCAN or louvain are available with RAPIDS, \
                               setting clusterer as DBSCAN!")
                self.clusterer = 'DBSCAN'
                if self.fromfile:
                    warnings.warn("Clusterer changed while loading paramiter file detected, \
                                   results may be inconsistent!")

            if self.metric_map != 'euclidean' or self.metric_clu != 'euclidean':
                warnings.warn("Only euclidean is available with RAPIDS, setting metrics \
                               as euclidean!")
                self.metric_map = 'euclidean'
                self.metric_clu = 'euclidean'
                if self.fromfile:
                    warnings.warn("Metrics changed while loading paramiter file detected, \
                                   results may be inconsistent!")

        """ Try to load parameters data. """

        if self.fromfile is not None:

            try:

                if isinstance(self.fromfile, str):
                    self.fromfile = self.interface.df.read_csv(self.fromfile)
                    self.fromfile.set_index('name', inplace=True)
                    self.fromfile.index = [
                        x.strip('cluster ') for x in self.fromfile.index]

                self.optimizer = 'grid'
                self.dynmesh = False

                if self._name not in self.fromfile.index:
                    #HERE TO START TO IMLPEMENT CONTINUE
                    self.nnei = []
                    self.ffrange = []
                    self.cparmrange = []
                    self.norm = self.interface.num.nan
                else:
                    self.dim = int(self.fromfile['dim'].loc[self._name])
                    self.neirange = [
                        int(self.fromfile['n_neighbours'].loc[self._name])]
                    self.ffrange = [
                        float(self.fromfile['genes_cutoff'].loc[self._name])]
                    self.cparmrange = [
                        float(self.fromfile['cluster_parm'].loc[self._name])]
                    self.metric_map = self.fromfile['metric_map'].loc[self._name]
                    self.metric_clu = self.fromfile['metric_clust'].loc[self._name]
                    self.norm = self.fromfile['norm'].loc[self._name]

                if self.interface.num.isnan(self.norm):
                    self.norm = None

            except BaseException:
                sys.exit('ERROR: there was a problem loading the parameters file.')
                raise

        """ Checks on optimizer choice. """

        if self.optimizer not in ['grid', 'de', 'auto']:
            sys.exit('ERROR: Optimizer must be either \'grid\' for Grid Search, \'de\' \
                      for Differential Evolution or \'auto\' for automatic selection.')

        if self.optimizer == 'de' and self.ffrange == 'kde':
            sys.exit('ERROR: KDE estimation of the low variance/MAD removal cutoff is \
                      not compatible with Differential Evolution.')

        if self.filterfeat not in ['variance', 'MAD', 'correlation', 'tSVD']:
            sys.exit('ERROR: Features filter must be either \'variance\' for low-variance removal, \
                      \'MAD\' for low-MAD removal, \'correlation\' for correlation removal, \
                      or \'tSVD\' for truncated SVD.')

        if self.optimizer == 'auto' and dynmesh == False:
            self.optimizer = 'grid'
            warnings.warn('Optimizer \'auto\' works only if dynamic mesh is active, \
                           falling to Grid Search')

        """ Checks optional modules. """

        if self.clusterer == 'HDBSCAN' and not OptionalImports.hdbscan:

            warnings.warn("HDBSCAN not found, setting clusterer as DBSCAN!")
            self.clusterer = 'DBSCAN'
            if self.fromfile:
                warnings.warn("Clusterer changed while loading paramiter file detected, \
                               results may be inconsistent!")

        """ Evaluate parameters granularity options. """

        if self.dynmesh:

            meshpoints = self.interface.num.rint(
                ((self.maxmesh - 1) * functions.sigmoid(
                    self.interface.num.log(
                        self.data_ix.shape[0]),
                    self.interface,
                    a=self.interface.num.log(500),
                    b=1))) + (
                self.minmesh)

            if self.optimizer == 'auto':
                if meshpoints**2 > 25:
                    self.optimizer = 'de'
                else:
                    self.optimizer = 'grid'

            if self.optimizer == 'grid':

                self.neipoints = meshpoints
                self.ffpoints = meshpoints

            elif self.optimizer == 'de':

                if self.minmesh <= 3:
                    self.minmesh = 4
                    meshpoints = self.interface.num.rint(
                        ((self.maxmesh - 1) * functions.sigmoid(
                            self.interface.num.log(
                                self.data_ix.shape[0]),
                            self.interface,
                            a=self.interface.num.log(500),
                            b=1))) + (
                        self.minmesh)

                self.depop = meshpoints
                self.deiter = meshpoints

        try:
            if isinstance(self.neipoints, list):
                self.neipoints = [int(x) for x in self.neipoints]
            else:
                self.neipoints = [int(self.neipoints)]
        except BaseException:
            sys.exit('ERROR: neipoints must be an integer or a list of integers')
            raise

        if self.optimizer == 'de':
            try:
                if isinstance(self.depop, list):
                    self.depop = [int(x) for x in self.depop]
                else:
                    self.depop = [int(self.depop)]
            except BaseException:
                sys.exit('ERROR: depop must be an integer or a list of integers')
                raise
            try:
                if isinstance(self.deiter, list):
                    self.deiter = [int(x) for x in self.deiter]
                else:
                    self.deiter = [int(self.deiter)]
            except BaseException:
                sys.exit('ERROR: deiter must be an integer or a list of integers')
                raise
            if self.ffrange == 'logspace':
                if self.filterfeat in ['variance', 'MAD', 'correlation']:
                    self.ffrange = [0.3, 0.9]
                if self.filterfeat == 'tSVD':
                    self.ffrange = [int(min([50,
                                             DataGlobal.dataset.loc[self.data_ix].shape[1] * 0.3])),
                                    int(DataGlobal.dataset.loc[self.data_ix].shape[1] * 0.9)]

        if self.optimizer == 'grid':
            try:
                if isinstance(self.ffpoints, list):
                    self.ffpoints = [int(x) for x in self.ffpoints]
                else:
                    self.ffpoints = [int(self.ffpoints)]
            except BaseException:
                sys.exit('ERROR: ffpoints must be an integer or a list of integers')
                raise
            if self.ffrange == 'logspace':
                if self.filterfeat in ['variance', 'MAD', 'correlation']:
                    self.ffrange = sorted([float(x) for x in self.interface.num.logspace(
                        self.interface.num.log10(0.3),
                        self.interface.num.log10(0.9),
                        num=self.ffpoints[0])])
                if self.filterfeat == 'tSVD':
                    self.ffrange = sorted([int(x) for x in self.interface.num.logspace(
                        self.interface.num.log10(min([50,
                            DataGlobal.dataset.loc[self.data_ix].shape[1] * 0.3])),
                        self.interface.num.log10(DataGlobal.dataset.loc[self.data_ix].shape[1] * 0.9),                             num=self.ffpoints[0])])

        """ Setup logging. """

        if self.debug:
            #logging.getLogger().setLevel(logging.DEBUG)
            logging.addLevelName(DEBUG_R, 'DEBUG_R')
            logging.getLogger().setLevel(DEBUG_R)
            os.environ['NUMBA_DEBUG'] = '0'  # Not working
            self._umap_rs = 32
            self._seed = 32
        else:
            logging.getLogger().setLevel(logging.INFO)
            self._umap_rs = None
            self._seed = None

    def _features_removal(self, cutoff):
        """ Either remove features with low variance/MAD, or high correlation
            from dataset according to a specified threshold (cutoff)
            or apply truncated SVD to reduce features to a certain number (cutoff).

        Args:
             cutoff (string or float): if filterfeat=='variance'/'MAD'/'correlation',
                percentage value for the low-variance/MAD/high-correlation removal cufoff,
                if 'kde' kernel density estimation will be used to find a single optimal
                low-variance/MAD/high-correlation cutoff;
                if filterfeat=='tSVD', dimensionality of the output data.

        Returns:
            (pandas dataframe): reduced-dimensionality input data.
            (tsvd object): trained tsvd instance, None if 'variance'/'MAD'/'correlation'.

        """

        if self.filterfeat == 'tSVD':

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

        if self.filterfeat in ['variance', 'MAD']:

            if cutoff == 'kde':
                new_data = functions._drop_min_KDE(
                    new_data, self.interface, type=self.filterfeat)
            else:
                new_data = functions._near_zero_var_drop(
                    new_data, self.interface, thresh=cutoff, type=self.filterfeat)

            logging.log(DEBUG_R, "Dropped Features #: " + '{:1.0f}'.format(
                DataGlobal.dataset.loc[self.data_ix].shape[1] - new_data.shape[1]))

            # Extra passage needed in case the transform data cut was
            # applied
            return DataGlobal.dataset.loc[self.data_ix][new_data.columns], None

        elif self.filterfeat == 'correlation':

            new_data = functions._drop_collinear(
                new_data, self.interface, thresh=cutoff)

            logging.log(DEBUG_R, "Dropped Features #: " + '{:1.0f}'.format(
                DataGlobal.dataset.loc[self.data_ix].shape[1] - new_data.shape[1]))

            # Extra passage needed in case the transform data cut was
            # applied
            return DataGlobal.dataset.loc[self.data_ix][new_data.columns], None

        else:

            sys.exit('ERROR: Oops, something went really wrong! Make sure filterfeat \
                        is properly set.\nIf you see this error please contact us on GitHub!')

    def _level_check(self):
        """ Stop the recursion if a given maxdepth parameter has been reached. """

        if self.maxdepth is not None:
            self.maxdepth -= 1
            if self.maxdepth == -1:
                self.clus_opt = None
                return True
            return False

    def _plot(self, n_nei, proj, cut_opt,
              keepfeat, decomposer, clus_opt, scoreslist):
        """ Produce a number of plots to visualize the clustering outcome at each stage of t
        he recursion.

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
                                    self.outpath)
        elif len(set(scoreslist[0])) > 1 and len(set(scoreslist[1])) == 1:
            plotting._plot_score([scoreslist[0],
                                 scoreslist[2]],
                                cut_opt,
                                'Features filter',
                                'scores_' + self._name,
                                self.outpath)
        elif len(set(scoreslist[1])) > 1 and len(set(scoreslist[0])) == 1:
            plotting._plot_score([scoreslist[1],
                                 scoreslist[2]],
                                n_nei,
                                'Nearest neighbours',
                                'scores_' + self._name,
                                self.outpath)

        """ Plot the Relative Pairwise Distance (RPD) distributions. """

        if self.RPD:
            #WARNING: unstable
            try:
                functions._calc_RPD(proj, clus_opt, self.interface,
                    True, self._name, self.outpath)
            except BaseException:
                logging.warning('RPD failed at step: ' + self._name)

        if self.filterfeat in ['variance','MAD'] \
            and isinstance(keepfeat,self.interface.df.Index):
            selcut = DataGlobal.dataset.loc[self.data_ix][keepfeat]

            """ Plots distribution of variance/MAD and low-variance/MAD genes cutoff. """

            plotting._plot_cut(self.interface.get_value(DataGlobal.dataset.loc[self.data_ix],
                              pandas=True),
                              self.interface.get_value(selcut, pandas=True),
                              'cut_' + self._name, self.outpath)

        elif self.filterfeat == 'tSVD' and decomposer is not None:
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
                    mapping.transform(scft), index=scft.index)
                # cudf workaround
                pj1.index = scft.index

                pj2 = self.interface.df.DataFrame(
                    mapping.transform(sct), index=sct.index)
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

            if self.savemap:
                with open(os.path.join(self.outpath,
                        'raccoon_data/' + self._name + '_2d.pkl'), 'wb') as file:
                    # keepfeat and decompt already in the not 2d map
                    pickle.dump(mapping, file)
                    file.close()

                proj.to_hdf(
                    os.path.join(self.outpath,
                        'raccoon_data/' + self._name + '_2d.h5'),
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
            self.outpath)

        """ Plot the same 2-dimensional umap with labels if provided. """

        if DataGlobal.labels is not None:

            plotting.plot_map(self.interface.get_value(proj, pandas=True), self.interface.get_value(
                DataGlobal.labels.loc[proj.index], pandas=True),
                'proj_labels_' + self._name, self.outpath)

        """ Plot the same 2-dimensional umap with transform only data if provided. """

        if self.transform is not None:
            transflab = self.interface.df.Series(
                'fit-transform', index=proj.index)
            transflab.loc[self.transform] = 'transform'
            plotting.plot_map(
                self.interface.get_value(proj,pandas=True),
                self.interface.get_value(transflab,pandas=True),
                'proj_trans_' + self._name,
                self.outpath)

    def _one_hot_encode(self, labs_opt, minpop=10):
        """ Construct and return a one-hot-encoded clusters membership dataframe.

        Args:
            labs_opt (pandas series): cluster membership series or list.

        Returns:
            tmplab (pandas dataframe): one-hot-encoded cluster membership dataframe.

        """

        if not isinstance(labs_opt, self.interface.df.DataFrame):
            labs_opt = self.interface.df.DataFrame(labs_opt)

        # cuml no sparse yet, bug inherited by cupy
        ohe = self.interface.one_hot(sparse=False)
        ohe.fit(labs_opt)

        tmplab = self.interface.df.DataFrame(
            ohe.transform(labs_opt),
            columns=self.interface.get_value(
                ohe.categories_[0])).astype(int)

        """ Discard clusters that have less than minpop of population. """

        tmplab.drop(tmplab.columns[self.interface.get_value(
            tmplab.sum() < minpop)], axis=1, inplace=True)
        # not_implemented_error: String Arrays is not yet implemented in cudf
        #tmplab = tmplab.set_index(DataGlobal.dataset.loc[self.data_ix].index.values)
        tmplab = tmplab.set_index(self.interface.get_value(
            #attempt to fix error for transform data
            #DataGlobal.dataset.loc[self.data_ix].index))
            labs_opt.index))
        
        """ Discard unassigned labels. """

        if -1 in tmplab.columns:
            tmplab.drop(-1,axis=1,inplace=True)

        """ Rename columns. """

        tmplab.columns = [self._name + "_" + str(x)
                          for x in range(len(tmplab.columns.values))]

        return tmplab

    def _elbow(self, pj):
        """ Estimates the point of flex of a pairwise distances plot.

        Args:
            pj (pandas dataframe/numpy matrix): projection of saxmples in the low-dimensionality space
                obtained with UMAP, or adjacency matrix if SNN.

        Returns:
            (float): elbow value.
        """

        if self.clusterer=='SNN':
            
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
                n_neighbors=2,
                metric=self.metric_clu,
                metric_params=self.mparams,
                n_jobs=-1).fit(pj)
            neigh = neigh.kneighbors(pj, return_distance=True)[0]
            
            if not isinstance(neigh, self.interface.df.DataFrame):
                neigh = self.interface.df.DataFrame(neigh)
            neigh.columns = ['0', 'elbow']
        
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

        if self.clusterer == 'louvain':
            logging.log(DEBUG_R, 
                'Resolution range guess: [{:.5f},{:.5f}]'.format(
                    0, 5))
            return self.interface.num.linspace(0, 5, 6)

        if self.clusterer  in ['DBSCAN', 'SNN']:
            ref = self._elbow(pj)
            logging.log(DEBUG_R, 
                'Epsilon range guess: [{:.5f},{:.5f}]'.format(
                    ref / 50, ref * 1.5))

            return self.interface.num.linspace(ref / 50, ref * 1.5, 100)

        elif self.clusterer == 'HDBSCAN':
            minbound = self.interface.num.amax(
                [self.minclusize, int(self.interface.num.sqrt(pj.shape[0] / 25))])
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
        """ Calculates Shared Nearest Neighbour (SNN) matrix

        Args:
            points (dataframe or matrix): points coordinates.
            num_neigh (int): number of neighbours considered
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

    def calc_score(self, points, labels):
        """ Select and calculate scoring function for optimization.

        Args:
            points (dataframe or matrix): points coordinates.
            labels (series or matrix): clusters assignment.

        Returns:
            (float): clustering score.

        """

        if len(self.interface.set(labels)) > len(points)-1:
            return -1

        if self.score == 'silhouette':
            return self.interface.silhouette(
                points, labels, metric=self.metric_clu)
        elif self.score == 'dunn':
            return self.interface.dunn(points, labels, metric=self.metric_clu)

        sys.exit('ERROR: score not recognized')

    def _find_clusters(self, pj, cparm, cse=None, algorithm=None):
        """ Runs the selected density-based clusters identification algorithm.

        Args:
            pj (dataframe or matrics): points coordinates.
            cparm (float): clustering parameter.
            cse (int): value of clustering_selection_epsilon for HDBSCAN.
            algorithm (string): value of algorithm for HDBSCAN.

        Returns:
            (list of int): list of assigned clusters. """


        if self.clusterer == 'SNN':
            return self.interface.cluster(pj,
                eps=cparm,
                min_samples=self.minclusize,
                metric='precomputed',
                n_jobs=-1,
                leaf_size=15)
        
        if self.clusterer == 'louvain':
            return self.interface.cluster_louvain(pj,
                resolution=cparm)

        if self.clusterer == 'DBSCAN':
            #NOTE: distances should be pre-computed like in SNN
            #to speed up this search
            return self.interface.cluster(pj,
                eps=cparm,
                min_samples=self.minclusize,
                metric=self.metric_clu,
                metric_params=self.mparams,
                n_jobs=-1,
                leaf_size=15)
            
        if self.clusterer == 'HDBSCAN':
            clusterer = HDBSCAN(
                algorithm=algorithm,
                alpha=1.0,
                approx_min_span_tree=True,
                gen_min_span_tree=False,
                leaf_size=15,
                allow_single_cluster=False,
                metric=self.metric_clu,
                metric_params=self.mparams,
                min_cluster_size=self.minclusize,
                min_samples=int(cparm),
                cluster_selection_epsilon=cse,
                p=None).fit(pj)
            return clusterer.labels_
        
        sys.exit('ERROR: clustering algorithm not recognized')

    def _run_single_instance(self, cutoff, nn):
        """ Run a single instance of clusters search for a given features cutoff and
        UMAP nearest neighbors number.

        Args:
            cutoff (float): features cutoff.
            nn (int): UMAP nearest neighbors value.

        Returns:
            sil_opt (float): silhoutte score corresponding to the best set of parameters.
            labs (pandas series): series with the cluster membership identified for each sample.
            cparm_opt (float): optimal clustering parameter value found.
            pj (pandas dataframe): low dimensionality data projection from UMAP.
            keepfeat (pandas index): set of genes kept after low /MAD removal, nan if 'tSVD'.
            decomposer (tsvd object): trained tsvd instance, None if 'variance'/'MAD'.

        """

        sil_opt = -0.0001
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

        if self.filterfeat in ['variance', 'MAD']:
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
                        self.supervised)),
                index=untransf.index)
            # cudf workaround
            pj.index = untransf.index

        else:

            pj = self.interface.df.DataFrame(
                mapping.fit_transform(
                    data_cut,
                    y=functions.loc_cat(
                        DataGlobal.labels,
                        data_cut.index,
                        self.supervised)),
                index=data_cut.index)
            # cudf workaround
            pj.index = data_cut.index

        if not pj.isnull().values.any():

            """ Set cluster_selection_epsilon for HDBSCAN. """

            cse = None
            hdbalgo = 'best'
            if self.clusterer == 'HDBSCAN':
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
 
            if self.clusterer == 'SNN':
                to_cluster=self.snn(pj,nn)
            elif self.clusterer == 'louvain':
                to_cluster=1-self.snn(pj,nn)
                if self.gpu:
                    to_cluster=self.interface.build_graph(to_cluster)
            else:
                to_cluster=pj

            """ Set clustering parameter range at the first iteration. """
            
            # TODO: check if better to update at every iteration
            cparmrange = self.cparmrange
            if cparmrange == 'guess':
                cparmrange = self._guess_parm(to_cluster)

            # Note: Calculating (H)DBSCAN on a grid of parameters is cheap even
            # with Differential Evolution.

            for cparm in cparmrange:  # self.cparmrange

                logging.log(DEBUG_R, 
                    'Clustering parameter: {:.5f}'.format(
                        self.interface.get_value(cparm)))

                labs = self._find_clusters(to_cluster, cparm, cse, hdbalgo)

                # not 100% sure about this, keep until weights on noise will be
                # added
                compset = self.interface.set(labs)
                compset.discard(-1)

                if len(compset) > 1:
                    sil = self.calc_score(pj, labs)
                else:
                    sil = 0

                if sil > sil_opt:
                    cparm_opt = cparm
                    sil_opt = sil
                    labs_opt=self.interface.df.Series(labs, index=pj.index)

        return sil_opt, labs_opt,\
            cparm_opt, pj, mapping, keepfeat, decomposer

    def _objective_function(self, params):
        """ Objective function for Differential Evolution.

        Args:
            params (list): a list containing a single feature cutoff and a
                UMAP nearest neighbors parameter.
        Returns:
            (float): loss value for the given set of parameters.
            labs (pandas series): series with the cluster membership identified
                for each sample.
            cparm_opt (float): optimal clustering parameter value found.
            pj (pandas dataframe): low dimensionality data projection from UMAP.
            keepfeat (pandas index): set of genes kept after low variance/MAD removal,
                nan if tSVD.
            decomposer (tsvd object): trained tsvd instance, None if 'variance'/'MAD'.

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
            sil_opt (float): Silhoutte score corresponding to the best set of parameters.
            labs_opt (pandas series): series with the cluster membership identified
                for each sample.
            cparm_opt (float): optimal clustering parameter value found.
            num_clus_opt (int):  total number of clusters determined by the search.
            nei_opt (int): optimal number of nearest neighbors used with UMAP.
            pj_opt (pandas dataframe): low dimensionality data projection from UMAP.
            cut_opt (float): optimal cutoff value used for the feature removal step
            keepfeat (pandas index): set of genes kept after low variance/MAD removal,
                nan if tSVD.
            decomp_opt (tsvd object): trained tsvd instance, None if 'variance'/'MAD'.
            scoreslist (list of float): list of all scores evaluated and their parameters.

        """

        sil_opt = -0.0001
        keepfeat = []
        decomp_opt = None
        labs_opt = [0] * DataGlobal.dataset.loc[self.data_ix].shape[0]
        cparm_opt = self.interface.num.nan
        nei_opt = DataGlobal.dataset.loc[self.data_ix].shape[0]

        if self.filterfeat in ['variance', 'MAD']:
            cut_opt = 1.0
        elif self.filterfeat == 'tSVD':
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

                # labs=None

                if self.transform is not None:

                    untransf = data_cut[~data_cut.index.isin(self.transform)]

                    pj = self.interface.df.DataFrame(
                        mapping.fit_transform(
                            untransf,
                            y=functions.loc_cat(
                                DataGlobal.labels,
                                untransf.index,
                                self.supervised)),
                        index=untransf.index)
                    # cudf workaround
                    pj.index = untransf.index

                else:

                    pj = self.interface.df.DataFrame(
                        mapping.fit_transform(
                            data_cut,
                            y=functions.loc_cat(
                                DataGlobal.labels,
                                data_cut.index,
                                self.supervised)),
                        index=data_cut.index)
                    # cudf workaround
                    pj.index = data_cut.index

                if not pj.isnull().values.any():

                    """ Set cluster_selection_epsilon for HDBSCAN. """

                    cse = None
                    hdbalgo = 'best'
                    if self.clusterer == 'HDBSCAN':
                        cse = float(self._elbow(pj))
                        if self.metric_clu == 'cosine':
                            hdbalgo = 'generic'
                            pj = pj.astype(self.interface.num.float64)

                    scoreslist[0].append(cutoff)
                    scoreslist[1].append(nn)
                    scoreslist[2].append(-0.0001)

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

                    if self.clusterer == 'SNN':
                        to_cluster=self.snn(pj,nn)
                    elif self.clusterer == 'louvain':
                        to_cluster=1-self.snn(pj,nn)
                        if self.gpu:
                            to_cluster=self.interface.build_graph(to_cluster)
                    else:
                        to_cluster=pj

                    """ Set clustering parameter range at the first iteration. """

                    cparmrange = self.cparmrange
                    if cparmrange == 'guess':
                        cparmrange = self._guess_parm(to_cluster)
                    # Note: Calculating (H)DBSCAN on a grid of parameters is
                    # cheap even with Differential Evolution.

                    for cparm in cparmrange:  # self.cparmrange

                        logging.log(DEBUG_R, 
                            'Clustering parameter: {:.5f}'.format(
                                self.interface.get_value(cparm)))
                        
                        labs = self._find_clusters(to_cluster, cparm, cse, hdbalgo)

                        # not 100% sure about this, keep until weights on noise
                        # will be added
                        compset = self.interface.set(labs)
                        compset.discard(-1)

                        if len(compset) > 1:
                            sil = self.calc_score(pj, labs)
                        else:
                            sil = 0

                        logging.log(DEBUG_R, 'Clustering score: {:.3f}'.format(sil))

                        if sil > scoreslist[2][-1]:
                            scoreslist[2].pop()
                            scoreslist[2].append(sil)

                        if sil > sil_opt:
                            cparm_opt = cparm
                            sil_opt = sil
                            pj_opt = pj
                            labs_opt = self.interface.df.Series(labs, index=pj_opt.index)
                            nei_opt = nn
                            pj_opt = pj
                            cut_opt = cutoff
                            map_opt = mapping
                            if self.filterfeat in [
                                    'variance', 'MAD', 'correlation']:
                                keepfeat = data_cut.columns
                            if self.filterfeat == 'tSVD':
                                decomp_opt = decomposer

        """ If an optimal solution was not found. """

        if sil_opt == -0.0001:
            logging.info('Optimal solution not found!')
            pj_opt = pj
            map_opt = mapping

        return sil_opt, labs_opt,\
               cparm_opt, nei_opt, pj_opt, cut_opt, map_opt, keepfeat, decomp_opt,\
               scoreslist

    def _KNN(self, nei_opt, pj_opt, labs_opt, cutoff=None):
        """ KNN classifier (REDUNDANT, TO REMOVE),
            updates membership assignment for transform-only data.

        Args:
            labs_opt (pandas series): series with the cluster membership identified
                for each sample.
            nei_opt (int): optimal number of nearest neighbors used with UMAP.
            pj_opt (pandas dataframe): low dimensionality data projection from UMAP.

        Returns:
            (pandas series): Series with the updated cluster membership identified
                for each sample.

        """

        missing = [i for i, x in enumerate(pj_opt.index)
                   if x not in labs_opt.index]

        mparams = {}
        if self.metric_clu == 'mahalanobis':
            try:
                mparams = {
                    'VI': self.interface.num.linalg.inv(
                        self.interface.num.cov(pj_opt.T))}
            except BaseException:
                mparams = {
                    'VI': self.interface.num.linalg.pinv(
                        self.interface.num.cov(pj_opt.T))}

        neigh = self.interface.n_neighbor(
            n_neighbors=len(missing) + nei_opt,
            metric=self.metric_map,
            metric_params=mparams,
            n_jobs=-1)
        neigh.fit(pj_opt)

        kn = neigh.kneighbors(pj_opt.iloc[missing], return_distance=True)
        kn = self.interface.num.array([[x, y] for x, y in zip(kn[0], kn[1])])
        mask = ~self.interface.num.isin(kn[:, 1], missing)
        # TODO: make prettier
        newk = [[kn[i, 0][mask[i]][1:nei_opt + 1], kn[i, 1]
                 [mask[i]][1:nei_opt + 1]] for i in range(len(kn))]

        clus_tmp = self._one_hot_encode(labs_opt)

        valals = []
        for k in self.interface.num.arange(len(newk)):
            vals = clus_tmp.loc[pj_opt.iloc[newk[k][1]].index].apply(
                lambda x: x / newk[k][0], axis=0)[1:]
            valals.append((vals.sum(axis=0) / vals.sum().sum()).values)

        knnlabs = self.interface.df.DataFrame(
            valals, index=pj_opt.index[missing], columns=clus_tmp.columns)

        if cutoff is not None:
            knnlabs[knnlabs < cutoff] == 0
            knnlabs[-1] = cutoff - .00001

        labs_new = knnlabs.idxmax(axis=1)

        return self.interface.df.concat([labs_opt, labs_new], axis=0)

    def _optimize_params(self):
        """ Wrapper function for the parameters optimization.

        Returns:
            sil_opt (float): Silhoutte score corresponding to the best set of parameters.
            labs_opt (pandas series): series with the cluster membership identified
                for each sample.
            cparm_opt (float): optimal clusters identification parameter value found.
            num_clus_opt (int):  total number of clusters determined by the search.
            nei_opt (int): optimal number of nearest neighbors used with UMAP.
            pj_opt (pandas dataframe): low dimensionality data projection from UMAP.
            cut_opt (float): optimal cutoff value used for the features removal step.
            keepfeat (pandas index): set of genes kept after low variance/MAD removal,
                nan if tSVD.
            decomp_opt (tsvd object): trained tsvd instance, None if 'variance'/'MAD'.
            reassigned (float): percentage of points forecefully assigned to a class
                if outliers='reassign'.
            scoreslist (list of float): list of all scores evaluated and their parameters.

        """

        logging.info(
            'Dimensionality of the target space: {:d}'.format(
                self.dim))
        logging.info('Samples #: {:d}'.format(
            DataGlobal.dataset.loc[self.data_ix].shape[0]))
        if self.dynmesh:
            if self.optimizer == 'grid':
                logging.info('Dynamic mesh active, number of grid points: {:d}'.format(
                        self.neipoints[0] * self.ffpoints[0]))
            if self.optimizer == 'de':
                logging.info('Dynamic mesh active, number of candidates: {:d} \
                              and iterations: {:d}'.format(
                        self.depop[0], self.deiter[0]))

        if self.transform is not None:
            logging.info('Transform-only Samples #: {:d}'.format(
                    len(self.transform)))
       
        numpoints = DataGlobal.dataset.loc[self.data_ix].shape[0]
        
        if self.transform is not None:
            numpoints -= len(self.transform)

        if numpoints != 0:
            
            if self.neirange == 'logspace':
                if self.neifactor >= 1:
                    minbound = self.interface.num.log10(
                        self.interface.num.sqrt(numpoints - 1))
                    maxbound = self.interface.num.log10(numpoints - 1)
                else:
                    minbound = self.interface.num.log10(
                        self.interface.num.sqrt(numpoints * self.neifactor))
                    maxbound = self.interface.num.log10(numpoints * self.neifactor)

                
                """ Neighbours cap. """

                if self.neicap is not None:
                    if minbound > self.interface.num.log10(self.neicap):
                        minbound = self.interface.num.log10(self.neicap / 10)
                    if maxbound > self.interface.num.log10(self.neicap):
                        maxbound = self.interface.num.log10(self.neicap)

                """ Hard limit. """

                if minbound < 1:
                    minbound = 1

                nnrange = sorted([int(x) for x in self.interface.num.logspace(
                    minbound, maxbound, num=self.neipoints[0])])

            elif self.neirange == 'sqrt':
                nnrange = [int(self.interface.num.sqrt(
                    numpoints * self.neifactor))]
            else:
                nnrange = self.neirange
                if not isinstance(nnrange, list):
                    nnrange = [nnrange]

            if self.neirange != 'logspace' and self.neicap is not None:
                nnrange = sorted(list(self.interface.set(
                    [x if x <= self.neicap else self.neicap for x in nnrange])))

        else:

            nnrange = []
        

        """ Run Optimizer. """

        if self.optimizer == 'grid':

            """ Grid Search. """
            logging.info('Running Grid Search...')

            sil_opt, labs_opt, cparm_opt, nei_opt, pj_opt, cut_opt, map_opt,\
            keepfeat, decomp_opt, scoreslist = self._run_grid_instances(
                nnrange)

            logging.info('Done!')

        elif self.optimizer == 'de':

            """ Differential Evolution. """

            logging.info('Running Differential Evolution...')

            # Note: this works as monodimensional DE, but may be slightly
            # inefficient
            bounds = [(min(self.ffrange), max(self.ffrange)),
                      (min(nnrange), max(nnrange))]
            sil_opt, labs_opt, cparm_opt, nei_opt, pj_opt, cut_opt, map_opt,\
            keepfeat, decomp_opt, scoreslist = de._differential_evolution(
                self._objective_function, bounds, maxiter=self.deiter[0],
                popsize=self.depop[0], integers=[False, True], seed=self._seed)

            if not isinstance(labs_opt,self.interface.df.Series):
                labs_opt=self.interface.df.Series(labs_opt, index=pj_opt.index)
            
            logging.info('Done!')
            
        else:
            sys.exit('ERROR: optimizer not recognized')
            
        if not isinstance(labs_opt,self.interface.df.Series):
            labs_opt=self.interface.df.Series(labs_opt, index=pj_opt.index)

        """ If data to be projected only is provided, apply projection. """

        if self.transform is not None:

            #if no clustering was found
            if len(keepfeat) == 0:
               labs_new=self.interface.df.Series([0]*len(self.transform),
                    index=self.transform)
               labs_opt=self.interface.df.concat([labs_opt,labs_new], axis=0)

            else:
                # A bit redundant, try to clean up
                if self.filterfeat in ['variance', 'MAD', 'correlation']:
                    transdata = DataGlobal.dataset[keepfeat].loc[self.transform]
                elif self.filterfeat == 'tSVD':
                    transdata = decomp_opt.transform(
                        DataGlobal.dataset.loc[self.transform])

                pj_opt = self.interface.df.concat([pj_opt, self.interface.df.DataFrame(
                    map_opt.transform(transdata), index=self.transform)], axis=0)

                logging.log(DEBUG_R, 
                    'Transform-only data found at this level: membership will be assigned with KNN')

                """ Assign cluster membership with k-nearest neighbors. """

                labs_opt = self._KNN(nei_opt, pj_opt, labs_opt)
        


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
                labs_opt = self._KNN(nei_opt, pj_opt, labs_out,
                                    cutoff=.5).loc[labs_opt.index]

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

    def recurse(self):
        """ Recursively clusters the input data, by first optimizing the parameters,
            binirizing the resulting labels, plotting and repeating. """

        self._depth += 1

        if self._level_check():
            return

        if DataGlobal.dataset.loc[self.data_ix].shape[0] < self.popcut:
            logging.info('Population too small!')
            self.clus_opt = None
            return

        minimum, clus_tmp, chosen, n_clu, n_nei, pj, cut, chomap, keepfeat,\
        decomp_opt, reassigned, scoreslist = self._optimize_params()

        """ Save cluster best parameters to table. """

        vals = ['cluster ' + self._name,
                DataGlobal.dataset.loc[self.data_ix].shape[0],
                n_clu, self.dim, minimum, n_nei,
                chosen, cut, self.metric_map,
                self.metric_clu, self.norm, reassigned]

        with open(os.path.join(self.outpath, 'raccoon_data/paramdata.csv'), 'a') as file:
            writer = csv.writer(file)
            writer.writerow(vals)
            file.close()

        """ Save intermediate UMAP data to be able to re-access specific clusters
        (expensive, only if debug True). """

        if self.savemap:
            with open(os.path.join(self.outpath,
                    'raccoon_data/' + self._name + '.pkl'), 'wb') as file:
                if self.filterfeat in ['variance', 'MAD', 'correlation']:
                    pickle.dump([keepfeat, chomap], file)
                elif self.filterfeat == 'tSVD':
                    pickle.dump([decomp_opt, chomap], file)
                file.close()
            pj.to_hdf(
                os.path.join(self.outpath,
                    'raccoon_data/' + self._name + '.h5'),
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
        clus_tmp = self._one_hot_encode(clus_tmp)

        """ Dig within each subcluster and repeat. """

        for l in list(clus_tmp.columns):

            sel_new = DataGlobal.dataset.loc[self.data_ix].loc[clus_tmp[clus_tmp[l] == 1].index]

            logging.info('Going deeper within Cluster # ' +
                         str(l) + ' [depth: {:d}'.format(self._depth) + ']')

            if self.transform is not None:
                indices = list(sel_new.index.values)
                to_transform = [x for x in indices if x in self.transform]
                if not to_transform:
                    to_transform = None
            else:
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
            if self.optimizer == 'grid' and len(self.neipoints) > 1:
                self.neipoints = self.neipoints[1:]
                logging.info(
                    'Parameters granilarity change ' +
                    '[nearest neighbours: {:d}'.format(
                        self.interface.get_value(self.neipoints[0])) + ']')
            if self.optimizer == 'de' and len(self.depop) > 1:
                self.depop = self.depop[1:]
                logging.info('Parameters granilarity change ' +
                             '[DE population: {:d}'.format(int(self.depop[0])) + ']')
            if self.optimizer == 'de' and len(self.deiter) > 1:
                self.deiter = self.deiter[1:]
                logging.info('Parameters granilarity change ' +
                             '[DE iterations: {:d}'.format(int(self.deiter[0])) + ']')

            deep = RecursiveClustering(sel_new.index, lab=None, transform=to_transform,
                dim=self.dim, epochs=self.epochs, lr=self.lr, neirange=self.neirange,
                neipoints=self.neipoints, neicap=self.neicap, metric_map=self.metric_map,
                metric_clu=self.metric_clu, popcut=self.popcut, filterfeat=self.filterfeat,
                ffrange=self.ffrange, ffpoints=self.ffpoints, optimizer=self.optimtrue,
                depop=self.depop, deiter=self.deiter, score=self.score, norm=self.norm,
                dynmesh=self.dynmesh, maxmesh=self.maxmesh, minmesh=self.minmesh,
                clusterer=self.clusterer, cparmrange=self.cparmrange, minclusize=self.minclusize,
                outliers=self.outliers, fromfile=self.fromfile, name=str(l), debug=self.debug,
                maxdepth=self.maxdepth, savemap=self.savemap, RPD=self.RPD,
                outpath=self.outpath, depth=self._depth, gpu=self.gpu, _user=False)

            deep.recurse()

            if deep.clus_opt is not None:

                # for now join not available in cudf
                #clus_tmp = self.interface.df.concat([clus_tmp, deep.clus_opt], axis=1, join='outer')
                deep.clus_opt = deep.clus_opt.reindex(clus_tmp.index)
                clus_tmp = self.interface.df.concat(
                    [clus_tmp, deep.clus_opt], axis=1)

                clus_tmp = clus_tmp.fillna(0)

                cols = list(clus_tmp.columns.values)
                for col in cols:
                    clus_tmp[col] = clus_tmp[col].astype(int)

        self.clus_opt = clus_tmp


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

    functions.setup(kwargs['outpath'], kwargs['RPD'])

    """ Run recursive clustering algorithm. """

    obj = RecursiveClustering(data, **kwargs)
    obj.recurse()

    """ Save the assignment to disk and buil tree. """

    tree = None
    if obj.clus_opt is not None:
        obj.clus_opt.to_hdf(
            os.path.join(
                kwargs['outpath'], 'raccoon_data/final_output.h5'),
            key='df')
        tree = trees.build_tree(
            obj.clus_opt, outpath=os.path.join(
                kwargs['outpath'], 'raccoon_data/'))

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
