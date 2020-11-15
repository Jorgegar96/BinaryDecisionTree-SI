import pandas as pd
from CrossValidation import ModelCrossValidation
from DecisionTree import DecisionTree
from PrunedTree import PrunedTree
import matplotlib.pyplot as plt
import numpy as np
import sys


def main():

    training_data = r'Datasets\training_data.csv'
    if len(sys.argv) > 1:
        training_data = f'{sys.argv[1]}'
    else:
        print("No argument found, the default training file will be used.")

    df = pd.read_csv(training_data)

    dt = DecisionTree()
    dt.trainModel(df, save_image=True)

    pt = PrunedTree(threshold=0.025)
    pt.pruneTree(r'Pickle Models\Unpruned_Tree.pickle', save_image=True)

    #multipleCrossValidation(df, 10000, 5, 10, 0.001, 0.05)


# Calculate sums for model result statistics and returns the sum and list of individual values
def calculateSum(model_results, k):
    sums = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}  # Stores accumulated sum
    indval = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}  # Stores individual values

    for i in range(k):
        indval['precision'].append(model_results[i][1]['precision'])
        sums['precision'] += indval['precision'][i]
        indval['accuracy'].append(model_results[i][1]['accuracy'])
        sums['accuracy'] += indval['accuracy'][i]
        indval['recall'].append(model_results[i][1]['recall'])
        sums['recall'] += indval['recall'][i]
        indval['f1'].append(model_results[i][1]['f1'])
        sums['f1'] += indval['f1'][i]

    return sums, indval


# Runs multiple instances of cross-validation in order to plot and compare hyper-parameter values
def multipleCrossValidation(df, sample_size=1600, k=5, laps=15, start=0.001, end=0.05, replace=False):
    threshold = np.linspace(start, end, laps)  # Linear space of threshold values
    acc = []  # Will be used to store average accuracy per threshold
    prec = []  # Will be used to store average precision per threshold
    rec = []  # Will be used to store average recall per threshold
    f1 = []  # Will be used to store average f1 per threshold

    acc_dict = {}  # Stores sample data corresponding to their accuracy
    rec_dict = {}  # Stores sample data corresponding to their accuracy
    prec_dict = {}  # Stores sample data corresponding to their accuracy
    f1_dict = {}  # Stores sample data corresponding to their accuracy

    for i in range(laps):
        print(f"Training lap #{i}")
        for j in range(int(sample_size / k)):
            cv = ModelCrossValidation()  # Instance to run cross-validation
            # model_results_u: unpruned, model_results_p: pruned
            model_results_u, model_results_p = cv.crossValidate(df, k, threshold[i])  # Cross-validate
            # We care only about de pruned models, since the unpruned don't use the threshold value
            sums, indval = calculateSum(model_results_p, k)  # Get sum of k values, and individual values as array
            if len(acc) == i:
                # Will append to add a value corresponding to the index i
                acc.append(sums['accuracy'])
                rec.append(sums['recall'])
                prec.append(sums['precision'])
                f1.append(sums['f1'])
                # Will assign create a new key for each dictionary and assign the indval array
                acc_dict[f"{threshold[i]}"] = indval['accuracy']
                rec_dict[f"{threshold[i]}"] = indval['recall']
                prec_dict[f"{threshold[i]}"] = indval['precision']
                f1_dict[f"{threshold[i]}"] = indval['f1']
            else:
                # Will accumulate sum values
                acc[i] += sums['accuracy']
                rec[i] += sums['recall']
                prec[i] += sums['precision']
                f1[i] += sums['f1']
                # Will append indval to the corresponding threshold value
                acc_dict[f"{threshold[i]}"] += indval['accuracy']
                rec_dict[f"{threshold[i]}"] += indval['recall']
                prec_dict[f"{threshold[i]}"] += indval['precision']
                f1_dict[f"{threshold[i]}"] += indval['f1']
        # Calculates the average over all the samples for the ith threshold value
        acc[i] /= sample_size
        rec[i] /= sample_size
        prec[i] /= sample_size
        f1[i] /= sample_size

    acc_df = pd.DataFrame(acc_dict)  # Stores sample data corresponding to their accuracy based on dictionaries
    rec_df = pd.DataFrame(rec_dict)  # Stores sample data corresponding to their recall based on dictionaries
    prec_df = pd.DataFrame(prec_dict)  # Stores sample data corresponding to their precision based on dictionaries
    f1_df = pd.DataFrame(f1_dict)  # Stores sample data corresponding to their f1-score based on dictionaries

    if not(replace):
        acc_prev_data = pd.read_csv(r'Model Performance\accuracy.csv')
        rec_prev_data = pd.read_csv(r'Model Performance\recall.csv')
        prec_prev_data = pd.read_csv(r'Model Performance\precision.csv')
        f1_prev_data = pd.read_csv(r'Model Performance\f1_score.csv')

        acc_df = pd.concat([acc_df,acc_prev_data], ignore_index=True)
        rec_df = pd.concat([rec_df, rec_prev_data], ignore_index=True)
        prec_df = pd.concat([prec_df, prec_prev_data], ignore_index=True)
        f1_df = pd.concat([f1_df, f1_prev_data], ignore_index=True)

    acc_df.to_csv(r'Model Performance\accuracy.csv')
    rec_df.to_csv(r'Model Performance\recall.csv')
    prec_df.to_csv(r'Model Performance\precision.csv')
    f1_df.to_csv(r'Model Performance\f1_score.csv')

    plotPerformance(threshold, acc, rec, prec, f1)


# Creates 4 plots corresponding to average measurements for each threshold p-value
def plotPerformance(threshold, acc, rec, prec, f1):
    fig, ax = plt.subplots(2, 2)
    fig.suptitle("Performance vs P-value")
    plt.rcParams.update({'font.size': 12})
    ax[0, 0].plot(threshold, acc, 'ro', markersize=1)  # Accuracy Plot
    ax[0, 0].set_title("Fig-1")
    ax[0, 0].set_xlabel("p-value")
    ax[0, 0].set_ylabel("Accuracy")
    ax[0, 1].plot(threshold, rec, 'bo', markersize=1)  # Recall Plot
    ax[0, 1].set_title("Fig-2")
    ax[0, 1].set_xlabel("p-value")
    ax[0, 1].set_ylabel("Recall")
    ax[1, 0].plot(threshold, prec, 'go', markersize=1)  # Precision Plot
    ax[1, 0].set_title("Fig-3")
    ax[1, 0].set_xlabel("p-value")
    ax[1, 0].set_ylabel("Precision")
    ax[1, 1].plot(threshold, f1, 'yo', markersize=1)  # F1 plot
    ax[1, 1].set_title("Fig-4")
    ax[1, 1].set_xlabel("p-value")
    ax[1, 1].set_ylabel("F1-Score")
    plt.show()


if __name__ == "__main__":
    main()
