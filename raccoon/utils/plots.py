"""
Plotting functions for RACCOON (Recursive Algorithm for Coarse-to-fine Clustering OptimizatiON)
F. Comitani     @2018-2020
A. Maheshwari   @2019
"""

import os
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


def _plotScore(scores, cparmrange, name='./scores.png', path=""):

    fig=plt.figure(figsize=(8,4))
    ax=plt.gca()
    maxs=[np.max(sc) for sc in scores]
    for i in range(len(scores)):
        if i==np.argmax(maxs):
            ax.plot(cparmrange, scores[i], linewidth=1.5, color='#ff7149', zorder=101, label='best parm set')
            ax.scatter(cparmrange[np.argmax(scores[i])], np.max(scores[i]), s=150, facecolor='none', color='#ff7149', zorder=101, lw=1.5)
        else:
            ax.plot(cparmrange, scores[i], linewidth=1, color='#AAAAAA', zorder=100)

    plt.tick_params(labelsize=15)
    plt.ylim([-1.1,1.1])
    plt.yticks(np.arange(-1,1.1,.25),np.arange(-1,1.1,.25))
    plt.xlabel('Clustering Parameter', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.legend(bbox_to_anchor=(1,1), fontsize=15)
    plt.tight_layout()

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
    

def plotViolin(vals, name='./rpdd.png', path=""):

    """ Generate a set of separate violin plot from given values.

    Args:
       vals (array of arrays of floats): Each internal array contains the values of a single violin plot.
       name (string): Name of output plot .png file.

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


    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(set(labels.values))))

    labels=labels.loc[df.index]

    plt.figure(figsize=(10,10))
    ax=plt.gca()
    for lab,col in zip(set(labels.values),colors):
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

