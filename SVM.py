from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier as ANN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
import math
import random
import pandas as pd


class classifier():
    def __init__(self,clfname):
        ''' load digits and iris datasets and partition
                        into training and testing sets, and create dataframe for classifier'''
        pd.set_option("display.max_columns", 500)
        df = pd.DataFrame()
        self.clfname =clfname


        digits = datasets.load_digits()
        n_samplesd = len(digits.images)
        raw_xd = digits.images.reshape((n_samplesd, -1))
        raw_yd = digits.target
        # shuffle the input and target
        X = raw_xd[raw_yd != 0, :]
        Y = raw_yd[raw_yd != 0]
        shuffled = list(zip(X, Y))
        random.shuffle(shuffled)
        d_X, d_y = zip(*shuffled)
        self.d_X_train = d_X[:math.floor(3 * n_samplesd / 4)]
        self.d_y_train = d_y[:math.floor(3 * n_samplesd / 4)]
        self.d_X_test = d_X[math.floor(3 * n_samplesd / 4):]
        self.d_y_test = d_y[math.floor(3 * n_samplesd / 4):]


        iris = datasets.load_iris()
        n_samplesi = len(iris.data)
        raw_xi = iris.data.reshape(n_samplesi, -1)
        raw_yi = iris.target
        # shuffle the input and target
        shuffled = list(zip(raw_xi, raw_yi))
        random.shuffle(shuffled)
        i_X, i_y = zip(*shuffled)
        self.i_X_train = i_X[:math.floor(3 * n_samplesi / 5)]
        self.i_y_train = i_y[:math.floor(3 * n_samplesi / 5)]
        self.i_X_test = i_X[math.floor(3 * n_samplesi / 5):]
        self.i_y_test = i_y[math.floor(3 * n_samplesi / 5):]
        self.i_X = i_X
        self.i_y = i_y

    def train(self):
        """create a grid search object that uses a classification SVM, a KNN estimator, an Artifical Neural Net,
        a decision tree, or an adaboosted decision tree as the backbone.
        the gridsearch object takes each dictionary in the params array and fits the estimator
        with all combinations within the dictionary, in addition, gridsearchcv also performs cross validation.
        refit is set to true, so the best combination of the parameters will be trained on the entire dataset"""
        if self.clfname == "SVM":
            # The parameters to be used for the SVM
            params = [{'kernel': ['linear']},
                      {'kernel': ['poly'], 'degree': [2,3,4]},
                      {'kernel': ['rbf'], 'gamma': [.1], 'C': [.01,1,100]}]
            # Create gridsearch objects for both Digits and Iris
            d_clf = GridSearchCV(SVC(), params, cv= 3, refit=True)
            i_clf = GridSearchCV(SVC(), params, cv= 3, refit= True)

            # fit both estimators and return the results of fitting each combination of parameters
            d_clf.fit(self.d_X_train,self.d_y_train)
            print(pd.DataFrame.from_dict(d_clf.cv_results_))
            i_clf.fit(self.i_X_train, self.i_y_train)
            print(pd.DataFrame.from_dict(i_clf.cv_results_))

            # return a tuple containing the trained estimators
            result_clf = (d_clf,i_clf)
            return result_clf

        if self.clfname == "KNN":
            # The parameters to be used for the nearest neighbor estimator
            params = {'n_neighbors': [3,5,9,15,21]}

            # Create 2 gridsearch objects for both digits and iris
            d_clf = GridSearchCV(KNN(weights='distance', algorithm='auto', n_jobs=-1), params, cv=3, refit=True)
            i_clf = GridSearchCV(KNN(weights='distance', algorithm='auto', n_jobs=-1), params, cv=3, refit=True)

            # fit both and print the results
            d_clf.fit(self.d_X_train, self.d_y_train)
            print(pd.DataFrame.from_dict(d_clf.cv_results_))
            i_clf.fit(self.i_X_train, self.i_y_train)
            print(pd.DataFrame.from_dict(i_clf.cv_results_))

            # Return the refit gridsearch objects as a tuple
            result_clf = (d_clf, i_clf)
            return result_clf

        if self.clfname == "ANN":
            # The parameters to use for the neural net
            params = [{'solver': ['sgd'], 'momentum': [0.0], 'learning_rate_init': [0.01, 0.1]},
                      {'solver': ['sgd'], 'momentum': [0.9], 'nesterovs_momentum': [True, False], 'learning_rate_init': [0.01, 0.1]},
                      {'solver': ['adam'], 'learning_rate_init': [0.01, 0.1], 'hidden_layer_sizes':
                          [(1, 100), (5, 100), (10, 100), (25, 100), (50, 100), (100, 100),
                           (150, 100), (200, 100), (300, 100), (500, 400), (1500, 400),
                           (2000, 400)]}]

            # create 2 gridsearch objects for both digits and iris
            d_clf = GridSearchCV(ANN(), params, cv=3, refit=True)
            i_clf = GridSearchCV(ANN(), params, cv=3, refit=True)

            # fit and print results for both
            d_clf.fit(self.d_X_train, self.d_y_train)
            print(pd.DataFrame.from_dict(d_clf.cv_results_))
            i_clf.fit(self.i_X_train, self.i_y_train)
            print(pd.DataFrame.from_dict(i_clf.cv_results_))

            # return the results as a tuple
            result_clf = (d_clf, i_clf)
            return result_clf

        if self.clfname == "DT":
            # The parameters for the decision tree
            params = [{'criterion': ['entropy'], 'max_features': [None, 'sqrt', 'log2'], 'min_samples_split': [3], 'max_depth': [1,5,10,15,None]}]

            # Create 2 gridsearch objects for both digits and iris
            d_clf = GridSearchCV(DT(), params, cv=3, refit=True)
            i_clf = GridSearchCV(DT(), params, cv=3, refit=True)

            # fit both and print the results
            d_clf.fit(self.d_X_train, self.d_y_train)
            print(pd.DataFrame.from_dict(d_clf.cv_results_))
            i_clf.fit(self.i_X_train, self.i_y_train)
            print(pd.DataFrame.from_dict(i_clf.cv_results_))

            # return a tuple of the results
            result_clf = (d_clf, i_clf)
            return result_clf

        if self.clfname == "Boost":
            # The parameters for the adabooster
            params = [{'n_estimators': [10,20,30,40,50,70]}]

            # create 2 decision tree objects for the adabooster
            d_dt = DT(criterion='entropy',min_samples_split=3, max_depth= 15, max_features= 'sqrt')
            i_dt = DT(criterion='entropy',min_samples_split=3, max_depth= 15, max_features= 'sqrt')

            # create 2 adabooster objects for digits and iris
            d_ada = AdaBoostClassifier(d_dt)
            i_ada = AdaBoostClassifier(i_dt)

            # create 2 gridsearch objects
            d_clf = GridSearchCV(d_ada,params)
            i_clf = GridSearchCV(i_ada,params)

            # fit and return results
            d_clf.fit(self.d_X_train, self.d_y_train)
            print(pd.DataFrame.from_dict(d_clf.cv_results_))
            i_clf.fit(self.i_X_train, self.i_y_train)
            print(pd.DataFrame.from_dict(i_clf.cv_results_))

            # Return the results as a tuple
            result_clf = (d_clf, i_clf)
            return result_clf

    def test(self, clf_tuple):
        """function takes in a tuple of the fit gridsearchcv objects for digits and iris.
        Predicts the using the withheld portions of the dataset.
        returns a pandas dataframe containing the classifier name, the input parameters, the error for both iris and digits,
        the time it took to fit the entire datasets, and the f1score"""
        d_clf, i_clf = clf_tuple
        d_y = d_clf.predict(self.d_X_test)
        print("Best Digits Estimator: ", d_clf.best_estimator_)
        d_score = metrics.accuracy_score(self.d_y_test, d_y)
        print("Score from best digits estimator: ", d_score)

        i_y = i_clf.predict(self.i_X_test)
        print("Best Iris Estimator: ", i_clf.best_estimator_)
        i_score = metrics.accuracy_score(self.i_y_test, i_y)
        print("Score from best iris estimator: ",i_score)

        return_dict = {'classifier': [self.clfname, self.clfname], "dataset": ['digits', 'iris'],
                       'error': [1 - d_score, 1-i_score], 'fit_time': [d_clf.refit_time_, i_clf.refit_time_],
                       'params': [d_clf.best_params_, i_clf.best_params_],
                       'F1':[metrics.f1_score(self.d_y_test, d_y, average='weighted'), metrics.f1_score(self.i_y_test, i_y, average='weighted')] }

        # return_dict = {'classifier': [self.clfname], 'Digits_Error': [1 - d_score], 'Iris_Error': [1 - i_score],
        #                'Digits_Time': [d_clf.refit_time_], 'Iris_Time': [i_clf.refit_time_],
        #                'Digits_Params': [d_clf.best_params_], 'Iris_Params': [i_clf.best_params_],
        #                'Digits_F1': metrics.f1_score(self.d_y_test, d_y, average='weighted'), 'Iris_F1': metrics.f1_score(self.i_y_test, i_y, average='weighted')}
        df = pd.DataFrame.from_dict(return_dict)
        print(df)
        return df

