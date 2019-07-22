from SVM import classifier
from plotter import plot_graph
import pandas as pd
import seaborn

def main():
    #call classifier for an svm
    clf = classifier("SVM")

    #call train for an svm
    result_clf = clf.train()

    #call test for an svm
    svm_stats = clf.test(result_clf)

    #create a dataframe object with the results from testing the svm
    df = pd.DataFrame.from_dict(svm_stats)

    #call classifier for KNN, train, test, and append the results to the dataframe
    clf = classifier("KNN")
    result_clf = clf.train()
    knn_stats = clf.test(result_clf)
    df = df.append(knn_stats,ignore_index=True)

    #call classifier for a neural net, train, test, and append results to the dataframe
    clf = classifier("ANN")
    result_clf = clf.train()
    ann_stats = clf.test(result_clf)
    df = df.append(ann_stats,ignore_index=True)

    #call classifier for a decision tree, train, test, and append results to the dataframe
    clf = classifier('DT')
    result_clf = clf.train()
    dt_stats = clf.test(result_clf)
    df = df.append(dt_stats, ignore_index= True)

    #call classifier for an adaboosted decision tree, train, test, and append results to dateframe
    clf = classifier('Boost')
    result_clf = clf.train()
    boost_stats = clf.test(result_clf)
    df = df.append(boost_stats, ignore_index=True)

    # Create plot_graph object with df to plot relevant graphs

    grapher = plot_graph(df)

    grapher.plot_graphs()


if __name__ == "__main__":
    main()