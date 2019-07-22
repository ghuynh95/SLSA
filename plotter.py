from matplotlib import pyplot as plt
import seaborn
import pandas as pd
class plot_graph():
    def __init__(self, data):
        pd.set_option("display.max_columns", 500)
        self.data = data

    def plot_graphs(self):
        """input a pandas dataframe and output graphs for """

        # Calculate the area created if error was plotted against training time for digit data set
        # to help analyze which is the more effective classifier for the dataset
        self.data['area'] = self.data['error'] * self.data['fit_time']

        # plot the error score for each classifier
        seaborn.catplot(x='classifier', y='error', hue='dataset', data=self.data, kind='bar')
        plt.show()
        seaborn.catplot(x='classifier', y='F1', hue='dataset', data=self.data, kind='bar')
        plt.show()
        seaborn.catplot(x='classifier', y='fit_time', hue='dataset', data=self.data, kind='bar')
        plt.show()