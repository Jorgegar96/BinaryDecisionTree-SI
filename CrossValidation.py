from DecisionTree import DecisionTree
from PrunedTree import PrunedTree
import random as rand
from math import sqrt

class ModelCrossValidation:

    def __init__(self):
        pass

    #Carries out a k-fold cross validation
    def crossValidate(self, dataset, k, threshold=0.05):

        #Gets and shuffles indices
        indices = list(dataset.index)
        rand.shuffle(indices)

        #Corrects for dataset sizes with remainders for a given k
        dataset_size = dataset.shape[0]
        dataset_size = dataset_size - (dataset_size % k)
        partition_size = dataset_size / k

        model_results_u, model_results_p = self.kFoldValidation(dataset, k, partition_size, indices, threshold)

        print("Unpruned Model Statistics")
        self.displayResults(model_results_u, k)

        print("Pruned Model Statistics")
        self.displayResults(model_results_p, k)

        return  model_results_u, model_results_p

    def kFoldValidation(self, dataset, k, partition_size, shuffled, threshold):
        test_data_file = r'Datasets\test_data.csv'
        tree_file = r'Pickle Models\unpruned.pickle'
        ptree_file = r'Pickle Models\pruned.pickle'
        model_results_u = []  # Results fro original tree
        model_results_p = []  # Results for pruned tree
        # Runs k-fold cross validation
        for i in range(k):
            print(f"Running fold #{i + 1} of {k}...")
            # Partitions dataset
            k_model_i = shuffled[int(partition_size) * i:int(partition_size) * (i + 1)]
            exclude = dataset.index.isin(k_model_i)
            dataset.loc[exclude].to_csv(test_data_file)

            # Run training routing on datasplit and validate
            dt = DecisionTree(test_data_file=test_data_file, tree_target_file=tree_file)
            conf_matrix, statistics = dt.trainModel(dataset.loc[~exclude], validate=True)
            model_results_u.append([conf_matrix, statistics])

            # Run pruning routine and validate
            pt = PrunedTree(test_data_file=test_data_file, tree_target_file=ptree_file, threshold=threshold)
            conf_matrix, statistics = pt.pruneTree(tree_file, validate=True)
            model_results_p.append([conf_matrix, statistics])

        return model_results_u, model_results_p

    # Displays statistical results of the k folds
    def displayResults(self, model_results, k):

        print("{:<6} {:<15} {:<15} {:<15} {:<15}".format('K', 'Accuracy', 'Precision', 'Recall', 'F1-Score'))
        print("-" * 66)

        averages = self.calculateAverage(model_results, k)

        print("-"*66)

        print("{:<6} {:<15} {:<15} {:<15} {:<15}".format(
            "AVG",
            round(averages['precision'], 4),
            round(averages['accuracy'], 4),
            round(averages['recall'], 4),
            round(averages['f1'], 4)
        ))

        # Calculates standard deviation
        std = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        for i in range(k):
            std['accuracy'] += (model_results[i][1]['accuracy'] - averages['accuracy'])**2
            std['precision'] += (model_results[i][1]['precision'] - averages['precision'])**2
            std['recall'] += (model_results[i][1]['recall'] - averages['recall'])**2
            std['f1'] += (model_results[i][1]['f1'] - averages['f1'])**2

        print("{:<6} {:<15} {:<15} {:<15} {:<15}".format(
            "STD",
            round(sqrt(std['precision'] / (k-1)), 4),
            round(sqrt(std['accuracy'] / (k-1)), 4),
            round(sqrt(std['recall'] / (k-1)), 4),
            round(sqrt(std['f1'] / (k-1)), 4)
        ))

    def calculateAverage(self, model_results, k):
        # Calculate averages and print individual values
        averages = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

        for i in range(k):
            prec = model_results[i][1]['precision']
            averages['precision'] += prec
            acc = model_results[i][1]['accuracy']
            averages['accuracy'] += acc
            recall = model_results[i][1]['recall']
            averages['recall'] += recall
            f1 = model_results[i][1]['f1']
            averages['f1'] += f1
            print("{:<6} {:<15} {:<15} {:<15} {:<15}".format(
                i,
                round(acc, 4),
                round(prec, 4),
                round(recall, 4),
                round(f1, 4))
            )

        averages['precision'] = averages['precision'] / k
        averages['accuracy'] = averages['accuracy'] / k
        averages['recall'] = averages['recall'] / k
        averages['f1'] = averages['f1'] / k

        return averages