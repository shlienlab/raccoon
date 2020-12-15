"""
Basic k-nearest neighbours classifier for RACCOON (Recursive Algorithm for Coarse-to-fine Clustering OptimizatiON)
F. Comitani     @2020
"""

import os
import pickle
import psutil

import pandas as pd
#from scipy.sparse import csr_matrix

import logging
import time

import raccoon.interface as interface

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



class knn:


    #TODO: add logging!

    """ To perform a basic distance-weighted k-nearest neighbours classification. """

    def __init__(self, data, oriData, oriClust, refpath="./raccoonData/", outpath="", root='0', debug=False, gpu=False):

        """ Initialize the the class.

        Args:
            data (matrix or pandas dataframe): Input data in pandas dataframe-compatible format (samples as row, features as columns).
            oriData (matrix or pandas dataframe): Original data clustered with RACCOON in pandas dataframe-compatible format (samples as row, features as columns).
            oriClust (matrix or pandas dataframe): Original RACCOON output one-hot-encoded class membership in pandas dataframe-compatible format 
                (samples as row, classes as columns).
            refpath (string): Path to the location where trained umap files (pkl) are stored (default, subdirectory racoonData of current folder).
            outpath (string): Path to the location where outputs will be saved (default, save to the current folder).
            root (string): Name of the root node, parent of all the classes within the first clustering leve. Needed to identify the appropriate pkl file (default 0)
            debug (boolean): Specifies whether algorithm is run in debug mode (default is False).
            gpu (bool): Activate GPU version (requires RAPIDS).
        """


        self.start_time = time.time()

        self.gpu = gpu

        """ Set up for CPU or GPU run """

        if self.gpu:
            try:
                self.interface=interface.interfaceGPU()
            except:
                warnings.warn("Warning: no RAPIDS found, running on CPU instead.")
                self.gpu=False

        if not self.gpu:
            self.interface=interface.interfaceCPU()


        if not isinstance(data, pd.DataFrame):
            try:
                data=pd.DataFrame(data)
            except:
                print('Unexpected error: ', sys.exc_info()[0])
                print('Input data should be in a format that can be translated to pandas dataframe!')
                raise

        if not isinstance(oriData, pd.DataFrame):
            try:
                data=pd.DataFrame(oriData)
            except:
                print('Unexpected error: ', sys.exc_info()[0])
                print('Input data (original) should be in a format that can be translated to pandas dataframe!')
                raise

        if not isinstance(oriClust, pd.DataFrame):
            try:
                data=pd.DataFrame(oriClust)
            except:
                print('Unexpected error: ', sys.exc_info()[0])
                print('Input data (clusters) should be in a format that can be translated to pandas dataframe!')
                raise


        self.oriData=oriData.astype(self.interface.num.float)
        self.data=data[self.oriData.columns].astype(self.interface.num.float)
        self.oriClust=oriClust
        self.refpath=refpath
        self.outpath=outpath
        self.root=root
        self.debug=debug

        self.children={}
        self.parents={}
        self._buildHierarchy()

        self.membership=[]

        """ Configure log. """

        logname='raccoon_knn_'+str(os.getpid())+'.log'
        print('Log information will be saved to '+logname)

        logging.basicConfig(level=logging.INFO, filename=os.path.join(outpath, logname), filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
        logging.getLogger('matplotlib.font_manager').disabled = True

        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            self._umapRs=32
        else:
            logging.getLogger().setLevel(logging.INFO)
            self._umapRs=None


    def _buildHierarchy(self):

        """ Builds a dictionary with information on the classess hierarchy. """

        # Unneccessarily complicated, but it works in case the classes have custom names
        # and are not already ordered hierarchically
        # TODO: clean up

        for i in range(len(self.oriClust.columns)):
            parent=self.oriClust.columns[i]
            parentIx=self.oriClust[self.oriClust[parent]==1].index
            self.children[parent]=[]
            for j in range(len(self.oriClust.columns)):
                if i!=j:
                    child=self.oriClust.columns[j]
                    childIx=self.oriClust[self.oriClust[child]==1].index
                    if child not in self.parents:
                        self.parents[child]=[]
                    if all(ix in parentIx for ix in childIx):
                        self.children[parent].append(child)
                        self.parents[child].append(parent)

        for parent,children in self.children.items():
            toremove=[]
            for j in children:
                for k in children:
                    if j!=k and j in self.children[k]:
                        toremove.append(j)
            self.children[parent]=[c for c in self.children[parent] if c not in toremove]

        for child,parents in self.parents.items():
            if parents!=[]:
                lengths=[self.oriClust[x].sum() for x in parents]
                self.parents[child]=parents[lengths.index(min(lengths))]
            else:
                self.parents[child]=None


    def _dampenChildProb(self):

        """ Renormalize the probabilities of a child class according to that of its parent. """

        for child in self.membership.columns:
            if self.parents[child] is not None:
                self.membership[child]*=self.membership[self.parents[child]]


    def assignMembership(self):

        """ Identifies class membership probabilities with a distance-weighted k-nearest neighbours algorith. """

        logging.info("Loading parameters data")

        names=[]

        paramdata=pd.read_csv(os.path.join(self.refpath,'paramdata.csv'))
        paramdata['name']=paramdata['name'].str.strip('cluster ')
        paramdata=paramdata.set_index('name',drop=True)

        for f in os.listdir(self.refpath): 

            if f.endswith('.pkl') and not f.endswith('_2d.pkl'):

                try:

                    with open(os.path.join(self.refpath,f), 'rb') as file:
                        names.append(f.strip('.pkl'))
                        loader=pickle.load(file)
                        genecut=loader[0]
                        mapping=loader[1]
                        nnei=mapping.n_neighbors
                        metric=paramdata['metric_clust'].loc[names[-1]]
                        norm=paramdata['norm'].loc[names[-1]]
                        file.close()

                except:

                    continue
                    
                logging.info("Working with subclusters of "+ names[-1])

                logging.debug('Nearest Neighbours #: {:d}'.format(nnei))
                logging.debug('Clustering metric: '+metric)

                if isinstance(genecut,pd.Index):
                    
                    """ low variance filter """

                    dfCut=self.data[genecut]
                
                else:
                
                    """ tSVD """
                    #sparseMat=csr_matrix(self.data.values)
                    #dfCut=pd.DataFrame(genecut.transform(sparseMat), index=self.data.index)
                    dfCut=pd.DataFrame(genecut.transform(self.data.values), index=self.data.index)



                if not self.interface.num.isnan(norm):

                    logging.debug('Norm: '+norm)
                
                    """ Normalize data. """

                    dfCut=pd.DataFrame(normalize(dfCut, norm=norm), index=dfCut.index, columns=dfCut.columns)

                proj=pd.DataFrame(mapping.transform(dfCut.values), index=dfCut.index)                

                if names[-1]==self.root:
                    refDf=self.oriData
                    nextClust=self.oriClust[[child for child,parent in self.parents.items() if parent is None]]
                else:
                    refDf=self.oriData[self.oriClust[names[-1]]==1]
                    nextClust=self.oriClust[self.oriClust[names[-1]]==1][self.children[names[-1]]]

                if isinstance(genecut,pd.Index):
                    
                    """ low variance filter """

                    dfCut=refDf[genecut]
                
                else:
                
                    """ tSVD """
                    #sparseMat=csr_matrix(refDf.values)
                    #dfCut=pd.DataFrame(genecut.transform(sparseMat), index=refDf.index)
                    dfCut=pd.DataFrame(genecut.transform(refDf.values), index=refDf.index)

                projRef=pd.DataFrame(mapping.transform(dfCut.values), index=dfCut.index)

                projAll=pd.concat([proj,projRef],axis=0)
               
                neigh=self.interface.nNeighbor(n_neighbors=nnei, metric=metric, n_jobs=-1).fit(projAll)
                kn=neigh.kneighbors(projAll, n_neighbors=len(projAll), return_distance=True)

                newk=[]
                for i in range(len(proj)):
                    newk.append([[],[]])
                    tupl=[(x,y) for x,y in zip(kn[0][i],kn[1][i]) if y in range(len(proj),len(projRef)+len(proj))]
                    for t in tupl:
                        newk[-1][0].append(t[0])
                        newk[-1][1].append(t[1])

                for k in range(len(newk)):
                    newk[k]=[newk[k][0][:nnei],newk[k][1][:nnei]]
                    
                valals=[]   
                for k in range(len(newk)):
                    vals=nextClust.loc[projAll.iloc[newk[k][1]].index].apply(lambda x: x/newk[k][0], axis=0)[1:]
                    valals.append((vals.sum(axis=0)/vals.sum().sum()).values)

                self.membership.append(pd.DataFrame(valals, index=proj.index, columns=nextClust.columns))     
            
        if len(names)>0:

            self.membership=pd.concat(self.membership,axis=1)
            self.membership=self.membership.reindex(columns=self.oriClust.columns)
            self.membership.fillna(0,inplace=True)

            #TODO: currently this assumes the classes are ordered hiearchically, make general
            self._dampenChildProb()

            logging.info('=========== Assignment Complete ===========')        
            logging.info('Total time of the operation: {:.3f} seconds'.format((time.time() - self.start_time)))
            logging.info(psutil.virtual_memory())    
        
        else:
            
            logging.error("No trained map files found!")
            print("ERROR: No trained map files found!")            


