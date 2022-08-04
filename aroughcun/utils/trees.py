"""
Hierarchical tree functions for RACCOON
F. Comitani     @2020-2022
"""

import os
from anytree import Node
from anytree.importer import JsonImporter, DictImporter
from anytree.exporter import JsonExporter

def build_tree(table, outpath=None):
    """ Set up a anytree object with useful information on
        the hierarchy of identified classes.

    Args:
        table (pandas dataframe): one-hot-encoded table of class membership.
        outpath (string): path where output files will be saved
            (includes filename).

    """

    nodes = []

    def find_parent(name, lista=nodes):
        parents = [l for l in lista if l.name == name[:-name[::-1].find('_')-1]]
        parents.append(None)
        return parents[0]

    nodes.append(Node('0', population=int(table.shape[0]),
                      parent=None, level=None, leaf=None))

    for col in table.columns:
        nodes.append(Node(col,
                          population=int(table[col].sum()),
                          parent=find_parent(col),
                          level=col.count('_'),
                          leaf=None))

    for n in nodes:
        n.leaf = len(n.children) == 0

    if outpath is not None:
        exporter = JsonExporter(indent=2, sort_keys=True)
        with open(outpath, 'w') as handle:
            exporter.write(nodes[0], handle)

    return nodes


def load_tree(file):
    """ Load an anytree object saved as json.

    Args:
        file (string): path to input json file.

    """

    importer = JsonImporter(DictImporter(nodecls=Node))
    with open(file, 'r') as handle:
        root = importer.read(handle)

    nodes = list(root.descendants)
    nodes = sorted(nodes, key=lambda x: int(x.name.split('\\')[-1]))

    return [root] + nodes
