from ModelTest import ModelTest
import sys


def main():
    testing_data = r'Datasets\testing_data.csv'
    if len(sys.argv) > 1:
        testing_data = f'{sys.argv[1]}'
    else:
        print("No argument found, the default testing file will be used.")

    print("Testing Original Model:")
    test = ModelTest()
    test.testModel(r'Pickle Models\Unpruned_Tree.pickle', testing_data, plot=True)

    print("Testing Pruned Model:")
    test = ModelTest()
    test.testModel(r'Pickle Models\Pruned_Tree.pickle', testing_data, plot=True)


if __name__ == "__main__":
    main()
