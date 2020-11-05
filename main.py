import pandas as pd
from CrossValidation import ModelCrossValidation
from DecisionTree import DecisionTree
from PrunedTree import PrunedTree
from ModelTest import ModelTest


def main():

    df = pd.read_csv('training_data.csv')
    '''
    dt = DecisionTree()
    dt.trainModel(df)

    test = ModelTest()
    test.testModel("Unpruned_Tree.pickle", "training_data.csv")


    pt = PrunedTree()
    pt.pruneTree("Unpruned_Tree.pickle")
    test = ModelTest()
    test.testModel("prunedTree.pickle", "training_data.csv")
    '''
    cv = ModelCrossValidation()
    cv.crossValidate(df, 10)
    #filt = df.index.isin([i for i in range(3,20)])
    #print(df.loc[~filt])



if __name__ == "__main__":
    main()