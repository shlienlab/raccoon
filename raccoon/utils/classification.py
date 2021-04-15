
"""
Basic k-nearest neighbours classifier for RACCOON
(Recursive Algorithm for Coarse-to-fine Clustering OptimizatiON)
F. Comitani     @2020
"""

import os
import sys
import pickle

import logging
DEBUG_R = 15

import time
import psutil

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import raccoon.interface as interface

class KNN:

    """ To perform a basic distance-weighted k-nearest neighbours classification. """

    def __init__(self, data, ori_data, ori_clust,
            refpath="./raccoon_data/", outpath="",
            root='0', debug=False, gpu=False):
        """ Initialize the the class.

        Args:
            data (matrix or pandas dataframe): input data in pandas dataframe-compatible format
                (samples as row, features as columns).
            ori_data (matrix or pandas dataframe): original data clustered with RACCOON in pandas
                dataframe-compatible format (samples as row, features as columns).
            ori_clust (matrix or pandas dataframe): original RACCOON output one-hot-encoded class
                membership in pandas dataframe-compatible format
                (samples as row, classes as columns).
            refpath (string): path to the location where trained umap files (pkl) are stored
                (default subdirectory racoon_data of current folder).
            outpath (string): path to the location where outputs will be saved
                (default save to the current folder).
            root (string): name of the root node, parent of all the classes within the first
                clustering leve. Needed to identify the appropriate pkl file (default 0).
            debug (boolean): specifies whether algorithm is run in debug mode (default is False).
            gpu (bool): activate GPU version (requires RAPIDS).
        """

        self.start_time = time.time()

        self.gpu = gpu

        """ Set up for CPU or GPU run. """

        if self.gpu:
            try:
                self.interface = interface.InterfaceGPU()
            except BaseException:
                warnings.warn("No RAPIDS found, running on CPU instead.")
                self.gpu = False

        if not self.gpu:
            self.interface = interface.InterfaceCPU()

        if not isinstance(data, self.interface.df.DataFrame):
            try:
                data = self.interface.df.DataFrame(data)
            except BaseException:
                print('Unexpected error: ', sys.exc_info()[0])
                print('Input data should be in a format that can be translated \
                       to pandas dataframe!')
                raise

        if not isinstance(ori_data, self.interface.df.DataFrame):
            try:
                ori_data = self.interface.df.DataFrame(ori_data)
            except BaseException:
                print('Unexpected error: ', sys.exc_info()[0])
                print('Input data (original) should be in a format that can be \
                       translated to pandas dataframe!')
                raise

        if not isinstance(ori_clust, self.interface.df.DataFrame):
            try:
                ori_clust = self.interface.df.DataFrame(ori_clust)
            except BaseException:
                print('Unexpected error: ', sys.exc_info()[0])
                print('Input data (clusters) should be in a format that can be \
                       translated to pandas dataframe!')
                raise

        self.ori_data = ori_data.astype(self.interface.num.float)
        self.data = data[self.ori_data.columns].astype(self.interface.num.float)
        self.ori_clust = ori_clust
        self.refpath = refpath
        self.outpath = outpath
        self.root = root
        self.debug = debug

        self.children = {}
        self.parents = {}
        self._build_hierarchy()

        self.membership = []

        """ Configure log. """

        logname = 'raccoon_knn_' + str(os.getpid()) + '.log'
        print('Log information will be saved to ' + logname)

        logging.basicConfig(
            level=logging.INFO,
            filename=os.path.join(outpath, logname),
            filemode="a+",
            format="%(asctime)-15s %(levelname)-8s %(message)s")
        logging.getLogger('matplotlib.font_manager').disabled = True

        if self.debug:
            #logging.getLogger().setLevel(logging.DEBUG)
            logging.addLevelName(DEBUG_R, 'DEBUG_R')
            logging.getLogger().setLevel(DEBUG_R)
            self._umap_rs = 32
        else:
            logging.getLogger().setLevel(logging.INFO)
            self._umap_rs = None

    def _build_hierarchy(self):
        """ Builds a dictionary with information on the classess hierarchy. """

        # Unneccessarily complicated, but it works in case the classes have custom names
        # and are not already ordered hierarchically
        # TODO: clean up

        for i in range(len(self.ori_clust.columns)):
            parent = self.ori_clust.columns[i]
            parent_ix = self.interface.get_value(self.ori_clust[self.ori_clust[parent] == 1].index)
            self.children[parent] = []
            for j in range(len(self.ori_clust.columns)):
                if i != j:
                    child = self.ori_clust.columns[j]
                    child_ix = self.interface.get_value(self.ori_clust[self.ori_clust[child] == 1].index)
                    if child not in self.parents:
                        self.parents[child] = []
                    if all(ix in parent_ix for ix in child_ix):
                        self.children[parent].append(child)
                        self.parents[child].append(parent)

        for parent, children in self.children.items():
            toremove = []
            for j in children:
                for k in children:
                    if j != k and j in self.children[k]:
                        toremove.append(j)
            self.children[parent] = [c for c in self.children[parent]
                                     if c not in toremove]

        for child, parents in self.parents.items():
            if parents != []:
                lengths = [self.ori_clust[x].sum() for x in parents]
                self.parents[child] = parents[lengths.index(min(lengths))]
            else:
                self.parents[child] = None

    def _dampen_child_prob(self):
        """ Renormalize the probabilities of a child class according to that of its parent. """

        for child in self.membership.columns:
            if self.parents[child] is not None:
                self.membership[child] *= self.membership[self.parents[child]]

    def assign_membership(self):
        """ Identifies class membership probabilities with a distance-weighted
            k-nearest neighbours algorith. """

        logging.info("Loading parameters data")

        names = []

        paramdata = self.interface.df.read_csv(
            os.path.join(self.refpath, 'paramdata.csv'))
        paramdata['name'] = paramdata['name'].str.strip('cluster ')
        paramdata = paramdata.set_index('name', drop=True)

        for f in os.listdir(self.refpath):

            if f.endswith('.pkl') and not f.endswith('_2d.pkl'):

                try:

                    with open(os.path.join(self.refpath, f), 'rb') as file:
                        names.append(f.strip('.pkl'))
                        loader = pickle.load(file)
                        genecut = loader[0]
                        mapping = loader[1]
                        nnei = mapping.n_neighbors
                        metric = paramdata['metric_clust'].loc[names[-1]]
                        norm = paramdata['norm'].loc[names[-1]]
                        file.close()

                except BaseException:

                    continue

                logging.info("Working with subclusters of " + names[-1])

                logging.log(DEBUG_R, 'Nearest Neighbours #: {:d}'.format(nnei))
                logging.log(DEBUG_R, 'Clustering metric: ' + metric)

                try:
                #if isinstance(genecut, self.interface.df.Index):

                    """ low information filter. """

                    df_cut = self.data[genecut]

                except:

                    """ tSVD. """
                    # sparse_mat=csr_matrix(self.data.values)
                    #df_cut=self.interface.df.DataFrame(genecut.transform(sparse_mat),
                    #iidex=self.data.index)
                    df_cut = self.interface.df.DataFrame(genecut.transform(self.data.values),
                            index=self.data.index)

                try:
                #if not self.interface.num.isnan(norm):

                    logging.log(DEBUG_R, 'Norm: ' + norm)

                    """ Normalize data. """

                    df_cut = self.interface.df.DataFrame(normalize(df_cut, norm=norm),
                        index=df_cut.index,
                        columns=df_cut.columns)
                
                except:

                    pass

                proj = self.interface.df.DataFrame(mapping.transform(df_cut.values),
                    index=df_cut.index)
                # cudf workaround
                proj.index = df_cut.index

                if names[-1] == self.root:
                    ref_df = self.ori_data
                    next_clust = self.ori_clust[[
                        child for child, parent in self.parents.items() if parent is None]]
                else:
                    ref_df = self.ori_data[self.ori_clust[names[-1]] == 1]
                    next_clust = self.ori_clust[self.ori_clust[names[-1]]== 1]\
                                [self.children[names[-1]]]

                try:
                #if isinstance(genecut, self.interface.df.Index):

                    """ low information filter. """

                    df_cut = ref_df[genecut]

                except:

                    """ tSVD. """
                    # sparse_mat=csr_matrix(ref_df.values)
                    #df_cut=self.interface.df.DataFrame(genecut.transform(sparse_mat),
                    #      index=ref_df.index)
                    df_cut = self.interface.df.DataFrame(
                        genecut.transform(ref_df.values), index=ref_df.index)

                proj_ref = self.interface.df.DataFrame(
                    mapping.transform(df_cut.values), index=df_cut.index)
                # cudf workaround
                proj_ref.index = df_cut.index

                proj_all = self.interface.df.concat([proj, proj_ref], axis=0)

                neigh = self.interface.n_neighbor(
                    n_neighbors=nnei, metric=metric, n_jobs=-1).fit(proj_all)
                kn = neigh.kneighbors(
                    proj_all,
                    n_neighbors=len(proj_all),
                    return_distance=True)

                newk = []
                for i in range(len(proj)):
                    newk.append([[], []])
                    tupl = [(x, y)
                            for x, y in zip(self.interface.get_value(kn[0][i]), 
                                            self.interface.get_value(kn[1][i]))
                            if y in range(len(proj), len(proj_ref) + len(proj))]
                    for t in tupl:
                        newk[-1][0].append(t[0])
                        newk[-1][1].append(t[1])

                for k in range(len(newk)):
                    newk[k] = [newk[k][0][:nnei], newk[k][1][:nnei]]

                valals = []
                for k in range(len(newk)):
                    #apply not available in cudf...
                    #vals = next_clust.loc[proj_all.iloc[newk[k][1]].index].apply(
                    #    lambda x: x / newk[k][0], axis=0)[1:]
                    
                    tmp = next_clust.loc[proj_all.iloc[newk[k][1]].index]

                    if not self.gpu:

                        vals = tmp.div(newk[k][0],axis=0)

                    else:
                        tmp.reset_index(drop=True, inplace=True)
                        vals = tmp.T.div(newk[k][0]).T

                    valals.append((vals.sum(axis=0) / vals.sum().sum()).values)
               
                valals=self.interface.num.stack(valals)

                self.membership.append(
                    self.interface.df.DataFrame(valals,
                        index=self.interface.get_value(proj.index),
                        columns=next_clust.columns))

        if len(names) > 0:

            self.membership = self.interface.df.concat(self.membership, axis=1)
            self.membership = self.membership.reindex(columns=self.ori_clust.columns)
            self.membership.fillna(0, inplace=True)

            # TODO: currently this assumes the classes are ordered
            # hiearchically, make general
            self._dampen_child_prob()

            logging.info('=========== Assignment Complete ===========')
            logging.info('Total time of the operation: {:.3f} seconds'.format(
                    (time.time() - self.start_time)))
            logging.info(psutil.virtual_memory())

        else:

            logging.error("No trained map files found!")
            print("ERROR: No trained map files found!")
