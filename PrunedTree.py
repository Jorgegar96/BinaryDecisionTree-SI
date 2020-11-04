from Node import Node
import pandas as pd
from scipy import stats
from GraphTree import GraphTree

class PrunedTree:

    def __init__(self):
        self.dtree = None
        self.orginal_tree = None

    '''BEGINNING OF TREE PRUNING CODE SECTION'''
    # Pruns the original tree
    def pruneTree(self, tree_file):

        # Reads the pickle file containing the original tree
        self.orginal_tree = pd.read_pickle(tree_file)

        self.recursivePruning(self.orginal_tree)
        self.dtree = self.orginal_tree

        pd.to_pickle(self.dtree, "PrunedTree.pickle")

        graph = GraphTree(self.dtree, 'prunedtree.png')
        graph.graphTree()

    def recursivePruning(self, tree):

        if not tree.children:
            return
        else:
            for child in tree.children:
                self.recursivePruning(child[0])

            if self.onlyLeafDescendants(tree):
                p = tree.statistics["p"]
                n = tree.statistics["n"]
                summation = 0
                for child in tree.children:
                    p_k = child[0].statistics["p"]
                    n_k = child[0].statistics["n"]
                    pk_hat = p / (p + n) * (p_k + n_k)
                    nk_hat = n / (p + n) * (p_k + n_k)
                    summation += (p_k - pk_hat)**2 / pk_hat + (n_k - nk_hat)**2 / nk_hat
                    #print(f"{p_k} - {n_k} vs {pk_hat} - {nk_hat}")
                #print(f"{tree.data}->{stats.chi2.pdf(summation, len(tree.children)-1)} - {summation} - {p} - {n} - {len(tree.children)-1}")
                #print(f"{stats.chi2.pdf(0.0,8)}")
                if stats.chi2.pdf(summation, len(tree.children)-1) > 0.05 or self.sameLabel(tree.children):
                    self.pruneLeaves(tree)

    def sameLabel(self, children):
        same = True
        prev = children[0][0].data
        for child in children:
            same = same and (child[0].data == prev)
            prev = child[0].data
        return same

    def pruneLeaves(self, tree):
        tree.children = []
        p = tree.statistics["p"]
        n = tree.statistics["n"]
        pos_class = tree.statistics["pos_class"]
        neg_class = tree.statistics["neg_class"]
        tree.data = self.pluralityValue(p, n, pos_class, neg_class)

    # Returns the mayority class
    def pluralityValue(self, p, n, pos_class, neg_class):

        if p > n:
            return pos_class
        elif p < n:
            return neg_class
        else:
            return "50-50"


    def onlyLeafDescendants(self, tree):
        for child in tree.children:
            if child[0].children:
                return False
        return True

    '''ENDING OF TREE PRUNING CODE SECTION'''
