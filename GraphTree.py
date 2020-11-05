from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import pygraphviz as pvg
import networkx as nx
from Node import Node

class GraphTree:

    def __init__(self, tree, name):
        # Empty dataset slot
        self.tree = tree
        self.G = nx.DiGraph().to_undirected()
        self.name = name

    def graphTree(self):
        self.buildGraph(self.tree, None, None)
        self.drawTree(self.name)
    """BEGINNING OF MODEL DISPLAY CODE SECTION"""

    def buildGraph(self, tree, parent, flag):
        parent_name = ''
        if not (tree.data in nx.get_node_attributes(self.G, 'label')):
            parent_name = tree.data
            self.G.add_node(tree.data, label=tree.data+'\n'+'pos: '+str(tree.statistics["p"])+'\n'+'neg: '+str(tree.statistics["n"]))
        else:
            contador = 0
            for node in nx.get_node_attributes(self.G, 'label'):
                if tree.data in node:
                    contador += 1
            parent_name = f"{tree.data}{contador}"
            self.G.add_node(parent_name, label=tree.data+'\n'+'pos: '+str(tree.statistics["p"])+'\n'+'neg: '+str(tree.statistics["n"]))
        if parent != None:
            contador = 0
            for node in nx.get_node_attributes(self.G, 'label'):
                if tree.data in node:
                    contador += 1
            if contador > 1:
                name = f"{tree.data}{contador - 1}"
            else:
                name = tree.data
            self.G.add_edge(parent.data, name, label=flag, color='blue')
        for child in tree.children:
            temp = Node(parent_name)
            self.buildGraph(child[0], temp, child[1])

    def drawTree(self, name):
        # Graficacion del arbol

        # set defaults
        self.G.graph['graph'] = {'rankdir': 'TD'}
        self.G.graph['node'] = {'shape': 'rectangle'}
        A = to_agraph(self.G)
        A.layout('dot')
        A.draw(name)

    """ENDING OF MODEL DISPLAY CODE SECTION"""