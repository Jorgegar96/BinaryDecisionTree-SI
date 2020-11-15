import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import random as rand

class ModelTest:

    def __init__(self):
        self.pos_class = None
        self.neg_class = None

    """BEGINNING OF MODEL VALIDATION CODE SECTION"""

    #Runs a validation test for a particular tree model
    def testModel(self, tree_file, test_data_file, plot=False):

        test_set = pd.read_csv(test_data_file)
        dtree = pd.read_pickle(tree_file)
        self.pos_class = dtree.statistics["pos_class"]
        self.neg_class = dtree.statistics["neg_class"]

        predictions = self.runPrediction(test_set, dtree)

        confusion_matrix = self.buildConfusionMatrix(predictions, list(test_set['class']))
        if plot:
            self.plotConfusionMatrix(confusion_matrix)

        basic_statistics = self.basicModelStatistics(confusion_matrix)
        if plot:
            self.plotModelStatistics(basic_statistics)

        return confusion_matrix, basic_statistics


    # Runs a series of predictions based on a list of observations
    def runPrediction(self, test_data, model):
        predictions = []

        #Run predictions for each observation
        for row in test_data.index:
            predictions.append(self.predict(test_data.loc[row], model))

        return predictions

    # Determines an observations predicted label based on the generated tree
    def predict(self, observation, dtree):
        if not dtree.children:
           return self.determineLabel(dtree)
        else:
            for child in dtree.children:
                if observation[dtree.data] == child[1]:
                    return self.predict(observation, child[0])
                    break
        return "ERROR: Match not found"

    # Determines class label
    def determineLabel(self, dtree):
        if dtree.data == self.pos_class or dtree.data == self.neg_class:
            return dtree.data
        else:
            proportions = dtree.statistics["p"] / (dtree.statistics["p"]+dtree.statistics["n"])
            if rand.random() < proportions:
                return self.pos_class
            else:
                return self.neg_class

        # Builds Confusion Matrix
    def buildConfusionMatrix(self, predictions, train_labels):
        tp = 0  # True Positive counts
        fp = 0  # False Positive counts
        tn = 0  # True Negative counts
        fn = 0  # False Negative counts

        for index, prediction in enumerate(predictions):
            op = ''
            if prediction == self.pos_class and prediction == train_labels[index]:
                tp += 1
                op = "tp"
            elif prediction == self.pos_class and prediction != train_labels[index]:
                fp += 1
                op = "fp"
            elif prediction == self.neg_class and prediction == train_labels[index]:
                tn += 1
                op = "tn"
            elif prediction == self.neg_class and prediction != train_labels[index]:
                fn += 1
                op = "fn"
            #print(f"{prediction} -> {train_labels[index]} -> {op}")

        conf_matrix_dict = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

        return conf_matrix_dict

    #Plots confusion matrix using seaborn
    def plotConfusionMatrix(self, confusion_matrix_dict):

        # Print values in console
        for key, value in confusion_matrix_dict.items():
            print(f"{key}-> {value}")

        # Saves dictionary into matrix
        data = [
            [confusion_matrix_dict['tp'],confusion_matrix_dict['fp']],
            [confusion_matrix_dict['fn'],confusion_matrix_dict['tn']]
        ]

        # Creates dataframe and plots confusion matrix using Seaborn
        conf_m = pd.DataFrame(data, columns=['windows', 'linux'], index=['windows', 'linux'])
        conf_m.index.name = "Predicted"
        conf_m.columns.name = "Actual"
        plt.figure(figsize=(10,7))
        sb.set(font_scale=1.4)
        sb.heatmap(conf_m, cmap="Blues", annot=True, annot_kws={"size": 24}, fmt='d').set_title("Confusion Matrix")
        plt.show()

    # Returns the calculations of basic model performance statistics
    def basicModelStatistics(self, conf_matrix_dict):
        tp = conf_matrix_dict["tp"]
        fp = conf_matrix_dict["fp"]
        tn = conf_matrix_dict["tn"]
        fn = conf_matrix_dict["fn"]

        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall)
        '''
        if tp + fp != 0:
            precision = tp / (tp + fp)
            f1 = 2 * precision * recall / (precision + recall)
        else:
            precision = 0
            f1 = 0
        '''
        statistics_dict = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

        return statistics_dict

    #Plots table with basic performance statistics
    def plotModelStatistics(self, statistics):

        # Print values in console
        for key, value in statistics.items():
            print(f"{key}-> {value}")

        data = []
        for key, value in statistics.items():
            data.append([key, round(value, 4)])

        fig = plt.figure(dpi=80)
        ax = fig.add_subplot(1, 1, 1)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        #ax.figure(2)
        table = ax.table(cellText=data,loc='center' )
        table.set_fontsize(14)
        table.scale(1,4)
        ax.axis('off')
        plt.show()

    """ENDING OF MODEL VALIDATION CODE SECTION"""