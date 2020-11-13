import pandas as pd
from scipy import stats
from GraphTree import GraphTree
from ModelTest import ModelTest


class PrunedTree:

    def __init__(
            self,
            tree_target_file=r'Pickle Models\Pruned_Tree.pickle',
            image_target_file=r'Tree Images\decisionprunedtree.png',
            test_data_file=r'Datasets\training_data.csv',
            threshold=0.05
    ):
        self.dtree = None
        self.orginal_tree = None
        self.tree_target_file = tree_target_file
        self.image_target_file = image_target_file
        self.test_data_file = test_data_file
        self.threshold = threshold

    '''BEGINNING OF TREE PRUNING CODE SECTION'''
    # Pruns the original tree
    def pruneTree(self, tree_file, validate=False, save_image=False):

        # Reads the pickle file containing the original tree
        self.orginal_tree = pd.read_pickle(tree_file)

        self.recursivePruning(self.orginal_tree)
        self.dtree = self.orginal_tree

        pd.to_pickle(self.dtree, self.tree_target_file)

        if save_image:
            graph = GraphTree(self.dtree, self.image_target_file)
            graph.graphTree()

        # Runs the statistics model validation
        if validate:
            test = ModelTest()
            return test.testModel(self.tree_target_file, self.test_data_file)

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
                if stats.chi2.pdf(summation, len(tree.children)-1) > self.threshold or self.sameLabel(tree.children):
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
