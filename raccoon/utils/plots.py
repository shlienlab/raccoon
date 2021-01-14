"""
Plotting functions for RACCOON (Recursive Algorithm for Coarse-to-fine Clustering OptimizatiON)
F. Comitani     @2018-2021
"""

import os
import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#TODO: make seaborn optional
import seaborn as sns
sns.set_style("darkgrid")

class Palettes:

    nupal=['#247ba0','#70c1b3','#b2dbbf','#f3ffbd','#ff7149']
    nupalmap=LinearSegmentedColormap.from_list('my_list', nupal, N=1000)

def _plotScore(scores, parmOpt, xlab, name='./scores.png', path=""):

    """ Plot optimization score through iterations and hilights the optimal choice..

    Args:
        scores (list of float): list of scores through the clustering parameter iterations.
        parmOpt (float): optimal parameter.
        xname (string): x axis label.
        name (string): Name of resulting .png file.
        path (string): Path where output pictures should be saved.
    """

    fig=plt.figure(figsize=(8,4))
    ax=plt.gca()

    ax.plot(sorted(scores[0]),[y for _,y in sorted(zip(scores[0],scores[1]), key=lambda pair: pair[0])], lw=1.5, color=Palettes.nupal[0], zorder=100)
    ax.scatter([parmOpt],[max(scores[1])], s=150, lw=1.5,  facecolor='none', edgecolor=Palettes.nupal[1], zorder=101)

    plt.tick_params(labelsize=15)
    plt.ylim([-1.1,1.1])
    plt.yticks(np.arange(-1,1.1,.25),np.arange(-1,1.1,.25))
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel('Silhouette score', fontsize=20)
    plt.tight_layout()

    if not name.endswith('.png'):
        name=name+'.png'
    plt.savefig(os.path.join(path,'raccoonPlots/' + name), dpi=600)
    plt.close()

def _plotScoreSurf(scores, parmOpt, name='./scores_surf.png', path=""):

    """ Plot parameters optimization surface.

    Args:
        scores (list of float): list of scores through the clustering parameter iterations.
        parmOpt (list of float): optimal parameters.
        name (string): name of resulting .png file.
        path (string): path where output pictures should be saved.
    """

    fig=plt.figure(figsize=(9,8))
    ax=plt.gca()

    print(scores[0])
    print(scores[1])
    print(scores[2])

    plt.tricontourf(scores[0], scores[1], scores[2], zorder=0, cmap=Palettes.nupalmap, levels=np.arange(0,1.1,.1), vmin=0, vmax=1., extend='min')
    cbar = plt.colorbar()
    cbar.set_label(r'Silhouette score', fontsize=20)
    cbar.ax.tick_params(size=0)
    cbar.set_ticks(np.arange(0,1.1,.1))
    cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.1,.1)], fontsize=15)

    for i in range(len(scores[0])):
        if scores[0]!=parmOpt:
            plt.scatter(scores[0][i],scores[1][i], s=25, color='#333333', alpha=.8, zorder=1, edgecolor='none')
    plt.scatter(parmOpt[0],parmOpt[1], s=200, facecolors='none', edgecolors='#333333', alpha=.6, lw=2, zorder=1)

    ax.set_aspect(1./ax.get_data_ratio())
    
    plt.grid(False)
    mag=10**(math.floor(math.log(max(scores[1]), 10)))
    plt.xlim([min(scores[0])-.05, max(scores[0])+.05])
    plt.ylim([min(scores[1])-mag/20, max(scores[1])+mag/20])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Features Filter', fontsize=20)
    plt.ylabel('Nearest Neighbors', fontsize=20)
    ax.set_aspect(1./ax.get_data_ratio())

    if not name.endswith('.png'):
        name=name+'.png'
    plt.savefig(os.path.join(path,'raccoonPlots/' + name), dpi=600)
    plt.close()


def _plotCut(df, df_cut, name='./geneCut.png', path=""):

    """ Plot variance distributions before and after of the low-variance removal step.

    Args:
        df (pandas dataframe): Original data before cutting low variance columns.
        df_cut (pandas dataframe): Data after cutting low variance columns.
        name (string): Name of resulting .png file.

    """

    stdevs=df.std()
    cumsumx=stdevs.sort_values(ascending=True).values
    cumsum=stdevs.apply(lambda x: x**2).sort_values(ascending=False).cumsum().values[::-1]
    cumsum=[x/cumsum[0] for x in cumsum]
    stdevs=stdevs.values
    stdevs_r=df[df.columns.difference(df_cut.columns)].std().values
    stdevs_k=df_cut.std().values

    fig=plt.figure(figsize=(8,4))
    ax=plt.gca()
    ax.plot(cumsumx, cumsum, color='#555555', label='cumulative')
    sns.distplot(stdevs, rug=False, label='all',ax=ax,hist=False)
    sns.distplot(stdevs_k, rug=True, label='keep',ax=ax,hist=False)
    sns.distplot(stdevs_r, rug=True, label='remove',ax=ax,hist=False)

    plt.tick_params(labelsize=15)
    plt.xlabel('StDev [log2(TPM+1)]', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.legend(bbox_to_anchor=(1,1), fontsize=15)
    plt.tight_layout()

    if not name.endswith('.png'):
        name=name+'.png'
    plt.savefig(os.path.join(path,'raccoonPlots/' + name), dpi=600)
    plt.close()
    

def plotViolin(vals,  name='./rpdd.png', path=""):

    """ Generate a set of separate violin plot from given values.

    Args:
       vals (array of arrays of floats): Each internal array contains the values of a single violin plot.
       name (string): Name of output plot .png file.
       path (string): Path where output pictures should be saved.
    """

    fig, ax = plt.subplots(1, len(vals), figsize=((len(vals)*2+1),5), sharey=False)

    stdev=[np.nanstd(v) for v in vals]

    customp=[sns.cubehelix_palette(121)[int(s*1.0/np.nanmax(stdev)*100)] if not np.isnan(s) else 0 for s in stdev]

    for i in range(len(vals)):
        sns.violinplot(ax=ax[i], data=vals[i], linewidth=0, width=0.8, color=customp[i])
        sns.violinplot(ax=ax[i], data=vals[i], linewidth=1, width=.0, color=customp[i])

        ticks=['clu #'+str(x) for x in np.arange(len(vals))]
        ax[i].set_xticklabels([ticks[i]])
        ax[i].tick_params(axis='y', which='major', labelsize=20)
        ax[i].tick_params(axis='x', which='major', labelsize=20)
        plt.tight_layout()


    ax[0].set_ylabel('CosHet', fontsize=30)

    if not name.endswith('.png'):
        name=name+'.png'
    plt.savefig(os.path.join(path, 'raccoonPlots/' + name), dpi=600)
    plt.close()


def plotMap(df, labels, name='./projection.png', path=""):

    """ Generate a 2-dimensional scatter plot with color-coded labels.

    Args:
        df (pandas dataframe): 2d input data. 
        labels (series): Label for each sample.
        name (string): Name of output plot .png file.
        path (string): Path where output pictures should be saved.
    """
    

    lbvals=set(labels.values)

    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(lbvals)))

    labels=labels.loc[df.index]

    plt.figure(figsize=(10,10))
    ax=plt.gca()
    for lab,col in zip(lbvals,colors):
        ax.scatter(df.loc[labels[labels==lab].index][0], df.loc[labels[labels==lab].index][1],\
                   c=[col], s=10, label=lab)

    ax.set_aspect('equal')
    plt.legend(bbox_to_anchor=(1,1), fontsize=15)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    plt.tight_layout()

    if not name.endswith('.png'):
        name = name+'.png'

    plt.savefig(os.path.join(path, 'raccoonPlots/' + name), dpi=600, bbox_inches='tight')
    plt.close()

