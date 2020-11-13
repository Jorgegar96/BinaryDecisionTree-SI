from math import log2 as log
import numpy as np
import pandas as pd
from ModelTest import ModelTest
from Node import Node
from GraphTree import GraphTree


class DecisionTree:

    def __init__(self,
                 tree_target_file=r'Pickle Models\Unpruned_Tree.pickle',
                 image_target_file=r'Tree Images\decisiontree.png',
                 test_data_file=r'Datasets\training_data.csv'
                 ):
        # Empty dataset slot
        self.dataset = None
        self.dtree = None
        self.pos_class = None
        self.neg_class = None
        self.tree_target_file = tree_target_file
        self.image_target_file = image_target_file
        self.test_data_file = test_data_file

    """BEGINNING OF MODEL TRAINING CODE SECTION"""

    def trainModel(self, dataset, validate=False, save_image=False):
        print("Beginning training on Dataset...")
        self.dataset = dataset

        # Get positive class value
        self.pos_class = np.unique(np.array(self.dataset['class']))[1]
        # Get negative class value
        self.neg_class = np.unique(np.array(self.dataset['class']))[0]

        # Train the Algorithm
        self.dtree = self.decisionTreeLearning(self.dataset.copy(), self.dataset.copy())

        # Saves the tree in a pickle file
        pd.to_pickle(self.dtree, self.tree_target_file)

        if save_image:
            self.printTree(self.dtree, 0)

            # Creates a visual representation of the Decision Tree
            graph = GraphTree(self.dtree, self.image_target_file)
            graph.graphTree()

        # Runs the statistics model validation
        if validate:
            test = ModelTest()
            return test.testModel(self.tree_target_file, self.test_data_file)

    # Entropy Calculation based on p and n disjunct observations
    def crossEntropy(self, p, n):
        q = p / (p + n)

        # Validate for values equal to 0
        # term_1 is q * log(q)
        if q != 0:
            term_1 = q * log(q)
        else:
            term_1 = 0

        # term_2 is q * log(q)
        if 1 - q != 0:
            term_2 = (1 - q) * log(1 - q)
        else:
            term_2 = 0

        # The value should be the result of computing -(q * log(q) + (1 - q) * log(1 - q))
        return - (term_1 + term_2)

    # Goal value calculation
    def Goal(self, attribute, p, n, class_mask, dataset):
        return self.crossEntropy(p, n) - self.Remainder(attribute, p + n, class_mask, dataset)

    # Gets unique values for an attribute
    def uniqueValues(self, attribute, dataset):
        pass

    # Remainder value calculation
    def Remainder(self, attribute, total, class_mask, dataset):

        # Attribute values to iterate
        values = np.array(self.dataset[attribute])
        # Summation value storage
        summation = 0

        for value in np.unique(values):
            # Mask to filter by specific attribute value
            attr_mask = dataset[attribute] == str(value)

            # Count positive and negative ocurrances
            p_k = dataset.loc[(attr_mask & class_mask), 'class'].agg('count')
            n_k = dataset.loc[attr_mask, 'class'].agg('count') - p_k
            if p_k == 0 and n_k == 0:
                summation += 0
            else:
                # Accumulate calculated values
                summation += (p_k + n_k) / total * self.crossEntropy(p_k, n_k)

        return summation

    # Determines p and n values
    def nodeStatistics(self, dataset):
        # Mask to filter by class
        class_mask = dataset['class'] == self.pos_class
        # Count positive and negative ocurrances
        p = dataset.loc[class_mask, 'class'].agg('count')
        n = dataset['class'].agg('count') - p
        return p, n, class_mask

    # Determines the best regressor to use to split the data
    def bestSplitCandidate(self, dataset):

        p, n, class_mask = self.nodeStatistics(dataset)

        max_attr = ""
        max_goal = -1
        # print("#####################################################################")
        for attribute in dataset.columns:
            goal = self.Goal(str(attribute), p, n, class_mask, dataset)
            # print(f"{str(attribute)} -> {goal}")
            if goal > max_goal and attribute != 'class':
                max_attr = str(attribute)
                max_goal = goal
        # print(f"{str(max_attr)} -> {max_goal}")
        # print("#####################################################################")
        return max_attr, p, n

    # Returns the mayority class
    def pluralityValue(self, dataset):

        p, n, _ = self.nodeStatistics(dataset)

        if p > n:
            return self.pos_class
        elif p < n:
            return self.neg_class
        else:
            return self.pos_class

    # Recursive Decision-Tree building
    def decisionTreeLearning(self, data, parent_data):
        if data.empty:
            # Return its parent's plurality value
            # print("RETURNING")
            p, n, _ = self.nodeStatistics(parent_data)
            return Node(self.pluralityValue(parent_data), p, n, self.pos_class, self.neg_class)
        elif len(data.index) == data.loc[(data['class'] == self.pos_class), 'class'].agg('count') or \
                len(data.index) == data.loc[~(data['class'] == self.pos_class), 'class'].agg('count'):
            # Return Classification
            # print("RETURNING")
            p, n, _ = self.nodeStatistics(data)
            return Node(list(data['class'])[0], p, n, self.pos_class, self.neg_class)
        elif data.shape[1] == 1:
            # Return its own plurality value
            # print("RETURNING")
            p, n, _ = self.nodeStatistics(data)
            return Node(self.pluralityValue(data), p, n, self.pos_class, self.neg_class)
        else:
            # Split the data based on the best split attribute
            split_attribute, p, n = self.bestSplitCandidate(data)
            tree = Node(split_attribute, p, n, self.pos_class, self.neg_class)
            for value in np.unique(np.array(self.dataset[split_attribute])):
                subtree = self.decisionTreeLearning(
                    data.loc[(data[split_attribute] == value), data.columns != split_attribute].copy(),
                    data.copy(),
                )
                tree.addChild(subtree, value)
            # print("RETURNING")
            return tree

    def printTree(self, tree, tabs):
        tab = "     "
        print(f"{tabs * tab}{tree.data}")
        for child in tree.children:
            self.printTree(child[0], tabs + 1)

    """ENDING OF MODEL TRAINING CODE SECTION"""
