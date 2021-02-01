"""
Hierarchical tree functions for RACCOON (Recursive Algorithm for Coarse-to-fine Clustering OptimizatiON)
F. Comitani     @2020
"""

import os
from anytree import Node, LevelOrderGroupIter, Walker
from anytree.importer import JsonImporter, DictImporter
from anytree.exporter import JsonExporter

def buildTree(table, outpath=None):
   
    """ Set up a anytree object with useful information on the hierarchy of identified classes.

    Args:
        table (pandas dataframe): on-hot-encoded table of class membership
        outpath (string): path where output files will be saved.
   
    """
    
    nodes=[]
    
    def findParent(name, lista=nodes):
        parents = [l for l in lista if l.name == name[:-2]]
        parents.append(None)
        return parents[0]
    
    nodes.append(Node('0', population = int(table.shape[0]), 
                      parent=None, level=None, leaf=None))
    for col in table.columns:
        nodes.append(Node(col, population = int(table[col].sum()),
                          parent=findParent(col), level=col.count('_'), leaf=None))

    for n in nodes:
        n.leaf = len(n.children) == 0

    exporter = JsonExporter(indent=2, sort_keys=True)
    with open(os.path.join(outpath, 'tree.json'),'w') as handle:
        exporter.write(nodes[0],handle)
              
    return nodes

                     
def loadTree(file):
    
    """ Load an anytree object saved as json.

    Args:
        file (string): path to input json file.
   
    """

    importer = JsonImporter(DictImporter(nodecls=Node))
    with open(file,'r') as handle:
        root=importer.read(handle)

    nodes=list(root.descendants)
    nodes=sorted(nodes, key=lambda x: int(x.name.split('\\')[-1]))

    return [root]+nodes        


