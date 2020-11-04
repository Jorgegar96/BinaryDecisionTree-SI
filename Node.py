class Node:

    def __init__(self, data, p=None, n=None, pos_class=None, neg_class=None):
        self.data = data
        self.statistics = {'p':p, 'n':n, 'pos_class':pos_class, 'neg_class':neg_class}
        self.children = []

    def addChild(self, child, value):
        self.children.append([child, value])
