"""
Utility functions for RACCOON
F. Comitani     @2018-2022
A. Maheshwari   @2019
"""

import os
import sys
import shutil
import warnings
import csv
import pickle

import logging

from scipy.stats import gaussian_kde
from scipy.stats import median_absolute_deviation as mad
from scipy.signal import argrelextrema

from raccoon.utils.plots import plot_violin

def sort_len_num(lista):
    """ Sort elements of a list by length first, then by numbers.

    Args:
        lista (list): the list to sort.

    Returns:
        (list): the sorted list.
    """
    return sorted(lista, key=lambda x: (len(x), x))

def _near_zero_var_drop(data, interface, thresh=0.99, type='variance'):
    """ Drop features with low variance/MAD based on a threshold after sorting them,
        converting to a cumulative function and keeping the 'thresh' % most variant features.

    Args:

        data (pandas dataframe): input pandas dataframe (samples as row, features as columns).
        interface (obj): CPU/GPU numeric functions interface.
        thresh (float): percentage threshold for the cumulative variance/MAD.
        type (string): measure of variability, to be chosen between
            variance ('variance') or median absolute deviation ('MAD').
    """

    if type == 'variance':
        v_val = data.var(axis=0).values
    elif type == 'MAD':
        v_val = data.apply(mad, axis=0).values

    cs = interface.df.Series(v_val).sort_values(ascending=False).cumsum()

    #remove = cs[cs > cs.iloc[-1]*thresh].index
    # return data.drop(data.columns[remove], axis=1)

    # check if order of columns matters
    keep = cs[cs <= cs.iloc[-1] * thresh].index
    if len(keep) == 0:
        keep = [cs.index[0]]

    # temorary workaround to cuDF bug
    # where we cannot slice with index (not iterable error)
    return data.iloc[:, interface.get_value(keep)]


def _drop_collinear(data, interface, thresh=0.75):
    """ Drop collinear features above the 'thresh' % of correlation.
        WARNING: very slow! Use tSVD instead!

    Args:

        data (pandas dataframe): input pandas dataframe (samples as row, features as columns).
        interface (obj): CPU/GPU numeric functions interface.
        thresh (float): percentage threshold for the correlation.
    """

    crmat = interface.df.DataFrame(
        interface.num.corrcoef(data.astype(float).T),
        columns=data.columns,
        index=data.columns)
    crmat.index.name = None
    crmat2 = crmat.where(
        interface.num.triu(interface.num.ones(crmat.shape),
            k=1).astype(interface.num.bool)).stack()
    crmat2 = crmat2.reset_index().sort_values(0, ascending=False)
    crmat2 = crmat2[crmat2[crmat2.columns[2]] > thresh]

    toremove = []
    while len(crmat2[crmat2.columns[2]]) > 0:
        a = crmat2[crmat2.columns[0]].iloc[0]
        b = crmat2[crmat2.columns[1]].iloc[0]
        meana = crmat.loc[a, crmat.columns.difference([a, b])].mean()
        meanb = crmat.loc[b, crmat.columns.difference([a, b])].mean()

        toremove.append([a, b][interface.num.argmax([meana, meanb])])

        crmat2 = crmat2[(crmat2[crmat2.columns[0]] != toremove[-1])
                        & (crmat2[crmat2.columns[1]] != toremove[-1])]

    return data.drop(toremove, axis=1)


def _drop_min_KDE(data, interface, type='variance'):
    """ Use kernel density estimation to guess the optimal cutoff for low-variance removal.

    Args:
        data (pandas dataframe): input pandas dataframe (samples as row, features as columns).
        interface (obj): CPU/GPU numeric functions interface.
        type (string): measure of variability, to be chosen between
            variance ('variance') or median absolute deviation ('MAD').

    """

    if type == 'variance':
        v_val = data.var(axis=0).values
    elif type == 'MAD':
        v_val = data.apply(mad, axis=0).values

    # x = interface.num.arange(interface.num.amin(v_val), interface.num.amax(v_val),
    #              (interface.num.amax(v_val)-interface.num.amin(v_val))/100)
    x = interface.num.linspace(
        interface.num.amin(v_val),
        interface.num.amax(v_val),
        100)
    kde = gaussian_kde(v_val, bw_method=None)
    y = kde.evaluate(x)

    imax = argrelextrema(y, interface.num.greater)[0]
    imin = argrelextrema(y, interface.num.less)[0]
    cutoff = None

    """ Take the last min before abs max. """

    absmax = interface.num.argmax(y[imax])

    if absmax > 0:
        cutoff = x[interface.num.amax(
            [xx for xx in imin if xx < imax[absmax]])]

    if cutoff is not None:
        cs = interface.df.Series(v_val, index=data.columns)
        remove = cs[cs < cutoff].index.values
        if len(remove) == 0:
            remove = [cs.index[0]]
        data = data.drop(remove, axis=1)

    return data


def _calc_RPD(mh, labs, interface, plot=True, name='rpd', path=""):
    """ Calculate and plot the relative pairwise distance (RPD) distribution for each cluster.
        See XXX for the definition.
        DEPRECATED: UNSTABLE, only works with cosine.

    Args:
        mh (pandas dataframe): dataframe containing reduced dimensionality data.
        labs (pandas series): clusters memebership for each sample.
        interface (obj): CPU/GPU numeric functions interface.
        plot (boolean): True to generate plot, saves the RPD values only otherwise.
        name (string): name of output violin plot .png file.

    Returns:
        vals (array of arrays of floats): each internal array represents the RPD values of the corresponding cluster #.

    """

    from sklearn.metrics.pairwise import cosine_similarity

    cosall = cosine_similarity(mh)
    if not isinstance(labs, interface.df.Series):
        labs = interface.df.Series(labs, index=mh.index)
    else:
        labs.index = mh.index

    csdf = interface.df.DataFrame(cosall, index=mh.index, columns=mh.index)
    csdf = csdf.apply(lambda x: 1 - x)

    lbvals = interface.set(labs.values)

    centroids = []
    for i in lbvals:
        centroids.append(mh.iloc[interface.num.where(labs == i)].mean())

    centroids = interface.df.DataFrame(
        centroids, index=lbvals, columns=mh.columns)
    coscen = cosine_similarity(centroids)
    coscen = interface.df.DataFrame(coscen, index=lbvals, columns=lbvals)
    coscen = coscen.apply(lambda x: 1 - x)
    #interface.num.fill_diagonal(coscen.values, 9999)

    vals = []
    for i in lbvals:
        if i != -1:

            matrix = csdf[labs[labs ==
                               i].index].loc[labs[labs == i].index].values

            siblings = lbvals
            siblings.remove(i)
            siblings.discard(-1)

            """ take only the upper triangle (excluding the diagonal)
                of the matrix to avoid duplicates. """

            vals.append([x * 1.0 / coscen[i].loc[list(siblings)].min() for x in
                         matrix[interface.num.triu_indices(matrix.shape[0], k=1)]])

    with open(os.path.join(path, 'rc_data/rinterface.df.pkl'), 'rb') as f:
        try:
            cur_maps = pickle.load(f)
        except EOFError:
            cur_maps = []

    for i in range(len(vals)):
        cur_maps.append([name + "_" + str(i), vals[i]])

    with open(os.path.join(path, 'rc_data/rinterface.df.pkl'), 'wb') as f:
        pickle.dump(cur_maps, f)

    if plot:
        plot_violin(vals, interface, name, path)

    return vals


def setup_log (out_path, suffix=''):
    """ Set up logging.

    Args:
        out_path (string): path where output files will be saved.
        suffix (string): suffix to add to the log file
    """

    logname = 'raccoon_' + str(os.getpid()) + suffix + '.log'
    print('Log information will be saved to ' + logname)

    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(
            out_path,
            logname),
        filemode="a+",
        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.getLogger('matplotlib.font_manager').disabled = True


def setup(out_path=None, paramdata=True, chk=False, RPD=False, suffix='', delete=True):
    """ Set up folders that are written to during clustering,
    	as well as a log file where all standard output is sent.
    	If such folders are already present in the path, delete them.

    Args:
        out_path (string): path where output files will be saved.
        paramdata (bool): if true create parameters csv table
            (default True).
        chk (bool): if true create checkpoints subdirectory
            (default False).
        RPD (bool): deprecated, if true created RPD distributions base pickle
            (default False).
        suffix (string): suffix to add to the log file
        delete (bool): if true delete folders if already present,
            user confirmation will always be required before deleting folders
            (default True).
    """

    """ Build folders and delete old data if present. """

    try:
        os.makedirs(os.path.join(out_path, 'rc_data'))
        if chk:
            os.makedirs(os.path.join(out_path, 'rc_data/chk'))
        os.makedirs(os.path.join(out_path, 'rc_plots'))
    except FileExistsError:
        warnings.warn('rc_data/rc_plots already found in path!')
        
        if delete:
            answer = None
            while answer not in ['y', 'yes', 'n', 'no']:
                answer = input(
                    "Do you want to delete the old folders? [Y/N]  ").lower()
            if answer.startswith('y'):
                shutil.rmtree(os.path.join(out_path, 'rc_data'))
                os.makedirs(os.path.join(out_path, 'rc_data'))
                if chk:
                    os.makedirs(os.path.join(out_path, 'rc_data/chk'))
                shutil.rmtree(os.path.join(out_path, 'rc_plots'))
                os.makedirs(os.path.join(out_path, 'rc_plots'))
            else:
                print('Please remove rc_data/plots manually or \
                       change output directory before proceeding!')
                sys.exit(1)
       
        else:
        
            #quite ugly, remember to clean up
            try:
                os.makedirs(os.path.join(out_path, 'rc_data'))
            except:
                pass

            if chk:
                try:
                    os.makedirs(os.path.join(out_path, 'rc_data/chk'))
                except:
                    pass
            try:
                os.makedirs(os.path.join(out_path, 'rc_plots'))
            except:
                pass

    """ Generate empty optimal paramaters table,
        to be written to at each iteration. """

    if paramdata:
        
        vals = ['name', 'n_samples', 'n_clusters',
            'dim', 'obj_function_score', 'n_neighbours',
            'cluster_parm', 'features_cutoff', 'metric_map',
            'metric_clust', 'norm', 'reassigned', 'seed']
       
        if not delete and os.path.isfile(os.path.join(out_path, 'rc_data/paramdata.csv')):
           shutil.copyfile(os.path.join(out_path, 'rc_data/paramdata.csv'),\
                      os.path.join(out_path, 'rc_data/paramdata.BAK_' + str(os.getpid()) + '.csv'))
        
        if not os.path.isfile(os.path.join(out_path, 'rc_data/paramdata.csv')):
            with open(os.path.join(out_path, 'rc_data/paramdata.csv'), 'w') as file:
                writer = csv.writer(file)
                writer.writerow(vals)
                file.close()

    if RPD:
    
        """ Generate empty calc_RPD distributions pickle,
            to be written to at each iteration. TO REMOVE """
    
        if not delete and os.path.isfile(os.path.join(out_path, 'rc_data/rpd.pkl')):
            os.rename(os.path.join(out_path, 'rc_data/rpd.pkl'),\
                      os.path.join(out_path, 'rc_data/rpd.BAK_' + str(os.getpid()) + '.pkl'))
        
        with open(os.path.join(out_path, 'rc_data/rpd.pkl'), 'wb') as file:
            empty = []
            pickle.dump(empty, file)
            file.close()

    """ Configure log. """

    setup_log(out_path, suffix)


def sigmoid(x, interface, a=0, b=1):
    """ Sigmoid function

    Args:
        x (float): position at which to evaluate the function
        interface (obj): CPU/GPU numeric functions interface.
        a (float): center parameter
        b (float): slope parameter

    Returns:
        (float): sigmoid function evaluated at position x
    """

    return 1 / (1 + interface.num.exp((x - a) * b))


def loc_cat(labels, indices, supervised):
    """ Selects labels in
        supervised UMAP and transform them to categories.

    Args:
        indices (array-like): list of indices.
        supervised (bool): True if running superived UMAP.
    Returns:
        (Series): sliced labels series as categories if it exists.
    """

    if labels is not None and supervised:
        try:
            return labels.loc[indices].astype('category').cat.codes
        except BaseException:
            warnings.warn("Failed to subset labels.")
    return None


def calc_score(points, labels, score, metric_clu, interface):
    """ Select and calculate scoring function for optimization.

    Args:
        points (dataframe or matrix): points coordinates.
        labels (series or matrix): clusters assignment.
        score (str): score type.
        metric_clu (str): metric to use in the scoring functions.
        interface (obj): CPU/GPU numeric functions interface.

    Returns:
        (float): clustering score.

    """

    if len(interface.set(labels)) > len(points)-1:
        return -1

    if score == 'silhouette':
        return interface.silhouette(
            points, labels, metric=metric_clu)
    elif score == 'dunn':
        return interface.dunn(points, labels, metric=metric_clu)
    else:
        if hasattr(score, '__call__'):
    
            """ User defined scoring function. """
            
            return score(points, labels, metric=metric_clu)
        
    sys.exit('ERROR: score not recognized')


def one_hot_encode(labs_opt, name, interface, min_pop=None, rename=True):
    """ Build and return a one-hot-encoded clusters membership dataframe.

    Args:
        labs_opt (pandas series): cluster membership series or list.
        name (str): parent cluster name.
        interface (obj): CPU/GPU numeric functions interface.
        min_pop (int): population threshold for clusters, if None, keep all
            produced clusters.
        rename (bool): rename columns expanding the parent cluster
            name (default True).

    Returns:
        tmplab (pandas dataframe): one-hot-encoded cluster membership dataframe.

    """

    tmpix = None
    if isinstance(labs_opt, interface.df.Series):
        tmpix = labs_opt.index
    
    if not isinstance(labs_opt, interface.df.DataFrame):
        labs_opt = interface.df.DataFrame(labs_opt)
        #ugly workaround for cudf
        if tmpix is not None:
            labs_opt.index = tmpix

    # cuml no sparse yet, bug inherited by cupy
    ohe = interface.one_hot(sparse=False)
    ohe.fit(labs_opt)

    tmplab = interface.df.DataFrame(
        ohe.transform(labs_opt),
        columns=interface.get_value(
            ohe.categories_[0])).astype(int)

    """ Discard clusters that have less than min_pop of population. """

    if min_pop is not None:
        tmplab.drop(tmplab.columns[interface.get_value(
            tmplab.sum() < min_pop)], axis=1, inplace=True)
    
    # not_implemented_error: String Arrays is not yet implemented in cudf
    #tmplab = tmplab.set_index(DataGlobal.dataset.loc[data_ix].index.values)
    tmplab = tmplab.set_index(interface.get_value(
        #attempt to fix error for transform data
        #DataGlobal.dataset.loc[data_ix].index))
        labs_opt.index))

    tmplab.index = labs_opt.index

    """ Discard unassigned labels. """

    if -1 in tmplab.columns:
        tmplab.drop(-1,axis=1,inplace=True)


    if rename:
        """ Rename columns. """
        
        tmplab.columns = [name + "_" + str(x)
                          for x in range(len(tmplab.columns.values))]

    return tmplab


def unique_assignment(tab, root, interface):
    """ Assigns samples to their maximum probability class-path along the hierarchy.
        Starting from a probability matrix.

    Args:
        tab (pandas dataframe): original cluster membership probabilities table.
        root (str): name of the root class.
        interface (obj): CPU/GPU numeric functions interface.

    Returns:
        tab (pandas dataframe): one-hot-encoded cluster membership dataframe.

    """

    for cl in [root]+list(tab.columns):

       children = [x for x in tab.columns if
            x.startswith(cl) and 
            x.count('_') == cl.count('_')+1 ]

       if len(children)>0:

          #knn already sorts this out by itself
          #if cl != root:
          #  tab[children] = tab[children].mul(tab[cl], axis=0)
          
          noise = tab[children][tab[children].sum(axis=1) == 0].index
          #assign_vec = tab[children].idxmax(axis=1)
          #cudf workaround
          tmp_tab = tab[children]
         
          #ridicolous cudf workaround
          assign_vec = interface.df.Series(tmp_tab.columns[
                interface.get_value(
                interface.num.argmax(
                interface.get_value(tmp_tab),axis=1))])
          #cudf workaround
          assign_vec.index = tmp_tab.index
          assign_vec.loc[noise] = -1

          #temporarily let's leave it like this, it won't work if the names
          #of the clusters have been changed with suffixes
          if interface.gpu:
            assign_vec = interface.df.Series([x.split('_')[-1] for x in interface.get_value(assign_vec)],
                index = assign_vec.index).astype(int)
            #assign_vec = assign_vec.astype(str).applymap(lambda x: x.split('_')[-1]).astype(int)
          else:
            assign_vec = assign_vec.astype(str).apply(lambda x: x.split('_')[-1]).astype(int)
          
          ohe = one_hot_encode(assign_vec, cl,
                        interface, rename=False)
          ohe.columns=[cl+'_'+str(c) for c in ohe.columns]
          ohe[[x for x in children if x not in ohe.columns]]=0

          tab[children] = ohe[children]

    return tab.astype(int)
