"""
Plotting functions for RACCOON
F. Comitani     @2018-2022
"""

import os
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import pandas as pd

mpl.use('Agg')

sns.set_style("darkgrid")

class Palettes:

    nupal = ['#247ba0', '#70c1b3', '#b2dbbf', '#f3ffbd', '#ff7149']
    nupalmap = LinearSegmentedColormap.from_list('my_list', nupal, N=1000)

    midpal = ['#F8B195', '#F67280', '#C06C84', '#6C5B7B', '#355C7D'][::-1]
    midpalmap = LinearSegmentedColormap.from_list('my_list', midpal, N=1000)

def _plot_score(scores, parm_opt, xlab, name='./scores.png', path=""):
    """ Plot optimization score through iterations and highlights the optimal choice.

    Args:
        scores (list of float): list of scores through the clustering
            parameter iterations.
        parm_opt (float): optimal parameter.
        xname (string): x axis label.
        name (string): name of resulting .png file.
        path (string): path where output pictures should be saved.
    """

    fig = plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.set_facecolor('white')
    
    cieling = max([max(scores[1]), 1.])
    
    #fix facecolor of colorbar ax

    ax.plot(sorted(scores[0]),
            [y for _,y in sorted(zip(scores[0],scores[1]),
                                 key=lambda pair: pair[0])],
            lw=1.5,
            color=Palettes.nupal[0],
            zorder=100)
    ax.scatter([parm_opt], [max(scores[1])], s=150, lw=1.5,
               facecolor='none', edgecolor=Palettes.nupal[1], zorder=101)

    plt.tick_params(labelsize=15)
    plt.ylim([0, cieling + .1])
    plt.yticks(np.linspace(0, cieling, 9), np.linspace(0, cieling, 9))
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel('Objective function', fontsize=20)
    plt.tight_layout()

    if not name.endswith('.png'):
        name = name + '.png'
    plt.savefig(os.path.join(path, 'rc_plots/' + name), dpi=600)
    plt.close()


def _plot_score_surf(scores, parm_opt, name='./scores_surf.png', path=""):
    """ Plot parameters optimization surface.

    Args:
        scores (list of float): list of scores through the clustering
            parameter iterations.
        parm_opt (list of float): optimal parameters.
        name (string): name of resulting .png file.
        path (string): path where output pictures should be saved.
    """

    fig = plt.figure(figsize=(9, 8))
    ax = plt.gca()
    
    ax.set_facecolor('white')

    cieling = max([max(scores[2]), 1.])

    plt.tricontourf(scores[0],
        scores[1],
        scores[2],
        zorder=0,
        cmap=Palettes.midpalmap,
        levels=np.linspace(0, cieling, 11),
        vmin=0,
        vmax=cieling,
        extend='min')
    cbar = plt.colorbar()
    cbar.set_label(r'Objective function', fontsize=20)
    cbar.ax.set_facecolor('white')
    cbar.ax.tick_params(size=0)
    cbar.set_ticks(np.linspace(0, cieling, 11))
    cbar.ax.set_yticklabels(['{:.1f}'.format(x)
                             for x in np.linspace(0, cieling, 11)], fontsize=15)

    for i in range(len(scores[0])):
        if scores[0] != parm_opt:
            plt.scatter(scores[0][i],
                scores[1][i],
                s=25, color='#333333',
                alpha=.8, zorder=1,
                edgecolor='none')
    plt.scatter(parm_opt[0],
        parm_opt[1],
        s=200, facecolors='none',
        edgecolors='#333333',
        alpha=.6, lw=2,
        zorder=1)

    ax.set_aspect(1. / ax.get_data_ratio())

    plt.grid(False)
    mag = 10**(math.floor(math.log(max(scores[1]), 10)))
    plt.xlim([min(scores[0]) - .05, max(scores[0]) + .05])
    plt.ylim([min(scores[1]) - mag / 20, max(scores[1]) + mag / 20])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Features Filter', fontsize=20)
    plt.ylabel('Nearest Neighbors', fontsize=20)
    ax.set_aspect(1. / ax.get_data_ratio())

    if not name.endswith('.png'):
        name = name + '.png'
    plt.savefig(os.path.join(path, 'rc_plots/' + name), dpi=600)
    plt.close()


def _plot_cut(df, df_cut, name='./gene_cut.png', path=""):
    """ Plot variance distributions before and after of the low-variance removal step.

    Args:
        df (pandas dataframe): original data before cutting low variance columns.
        df_cut (pandas dataframe): data after cutting low variance columns.
        name (string): name of resulting .png file.
    """

    stdevs = df.std()
    cumsumx = stdevs.sort_values(ascending=True).values
    cumsum = stdevs.apply(
        lambda x: x**2).sort_values(ascending=False).cumsum().values[::-1]
    cumsum = [x / cumsum[0] for x in cumsum]
    stdevs = stdevs.values
    stdevs_r = df[df.columns.difference(df_cut.columns)].std().values
    stdevs_k = df_cut.std().values

    fig = plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.plot(cumsumx, cumsum, color='#555555', label='cumulative')
    sns.distplot(stdevs, rug=False, label='all', ax=ax, hist=False)
    sns.distplot(stdevs_k, rug=True, label='keep', ax=ax, hist=False)
    sns.distplot(stdevs_r, rug=True, label='remove', ax=ax, hist=False)

    plt.tick_params(labelsize=15)
    plt.xlabel('st_dev [log2(TPM+1)]', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.legend(bbox_to_anchor=(1, 1), fontsize=15)
    plt.tight_layout()

    if not name.endswith('.png'):
        name = name + '.png'
    plt.savefig(os.path.join(path, 'rc_plots/' + name), dpi=600)
    plt.close()


def plot_violin(vals, name='./rpdd.png', path=""):
    """ Generate a set of separate violin plot from given values.

    Args:
        vals (array of arrays of floats): each internal array contains
            the values of a single violin plot.
        name (string): name of output plot .png file.
        path (string): path where output pictures should be saved.
    """

    fig, ax = plt.subplots(
        1, len(vals), figsize=(
            (len(vals) * 2 + 1), 5), sharey=False)

    stdev = [np.nanstd(v) for v in vals]

    customp = [sns.cubehelix_palette(121)[int(s * 1.0 / np.nanmax(stdev) * 100)]
               if not np.isnan(s) else 0 for s in stdev]

    for i in range(len(vals)):
        sns.violinplot(ax=ax[i], data=vals[i],
            linewidth=0, width=0.8, color=customp[i])
        sns.violinplot(ax=ax[i], data=vals[i],
            linewidth=1, width=.0, color=customp[i])

        ticks = ['clu #' + str(x) for x in np.arange(len(vals))]
        ax[i].set_xticklabels([ticks[i]])
        ax[i].tick_params(axis='y', which='major', labelsize=20)
        ax[i].tick_params(axis='x', which='major', labelsize=20)
        plt.tight_layout()

    ax[0].set_ylabel('cos_het', fontsize=30)

    if not name.endswith('.png'):
        name = name + '.png'
    plt.savefig(os.path.join(path, 'rc_plots/' + name), dpi=600)
    plt.close()


def plot_map(df, labels, name='./projection.png', path=""):
    """ Generate a 2-dimensional scatter plot with color-coded labels.

    Args:
        df (pandas dataframe): 2d input data.
        labels (series): label for each sample.
        name (string): name of output plot .png file.
        path (string): path where output pictures should be saved.
    """

    lbvals = set(labels.values)

    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(lbvals)))

    # ugly, find a better way
    #try:
    labels = labels.loc[df.index]
    #except:
    #    labels=labels.reset_index(drop=True)
    
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    ax.set_facecolor('white')
    plt.grid(color='#aaaaaa')
    plt.axis('off')

    for lab, col in zip(lbvals, colors):
        ax.scatter(df.loc[labels[labels == lab].index].iloc[:,0],
                   df.loc[labels[labels == lab].index].iloc[:,1],
                   c=[col], s=10, label=lab)

    ax.set_aspect('equal')
    plt.legend(bbox_to_anchor=(1, 1), fontsize=15)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])

    if not name.endswith('.png'):
        name = name + '.png'

    plt.savefig(os.path.join(path, 'rc_plots/' + name),
        dpi=600, bbox_inches='tight')
    plt.close()


def plot_homogeneity(df1, df2, name='./homogeneity.png', path=""):
    """ Plot a heatmap of the homogeneity score between two sets 
        of clusters.

    Args:
        df1 (pandas dataframe): first one-hot-encoded clusters 
            membership table.
        df2 (pandas dataframe): second one-hot-encoded clusters 
            membership table.
        name (string): name of output plot .png file.
        path (string): path where output pictures should be saved.
    """

    dfs = pd.concat([df1,df2], axis=1).fillna(0)

    hs=[]
    for i in dfs.columns:
        hs.append([])
        for j in dfs.columns:
            numerator = len(set(dfs[dfs[i]==1].index).intersection(dfs[dfs[j]==1].index))
            denominator = dfs[i].sum()
             
            if denominator == 0 or numerator == 0:
                hs[-1].append(0)
            else:
                hs[-1].append(len(set(dfs[dfs[i]==1].index).intersection(dfs[dfs[j]==1].index))/dfs[i].sum())
                

    hs=pd.DataFrame(hs, index=dfs.columns, columns=dfs.columns)
    mask=hs.applymap(lambda x: False if x>=0.01 else True)
    hs=hs.applymap(lambda x: int(x*100))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('white')
    
    sns.heatmap(hs, annot=True, 
        cmap=Palettes.midpalmap, cbar=False,  
        annot_kws={"size": 20},fmt='d', mask=mask, linewidths=.5)

    ax.tick_params(labelsize=20, length=0)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title('Homogeneity Score %\n(%samples from row found in col)', fontsize=25, pad=10)
    ax.set_aspect('equal')

    if not name.endswith('.png'):
        name = name + '.png'

    plt.savefig(os.path.join(path, 'rc_plots/' + name),
        dpi=600, bbox_inches='tight')
    plt.close()
