# Empirical Test of Berkeley Acceptance Data
# Data from https://stats.idre.ucla.edu/r/dae/logit-regression/

import pandas as pd
import numpy as np
import csv
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import classification_tree as ct
import random_forest as rf

# Import Berkeley acceptance data, create training and test datasets
accepted = pd.read_csv('Empirical/acceptance.csv', header=0)
X = np.array(accepted.iloc[:,1:])
y = np.array(accepted.iloc[:,0])

alphas = [0, 0.2, 0.4, 0.6, 0.8]
trees = [1, 2, 5, 10]

for i in range(3):
    for alpha in alphas:

        # Randomly split
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33)

        X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test), 
        y_train, y_test = pd.DataFrame(y_train), pd.DataFrame(y_test)
        
        # Create decision tree
        tree = ct.DecisionTree(X_train, y_train)
        tree.prune(alphas=[alpha], cross_validate=False)
        preds = tree.predict(X_test)
        print("Confusion matrix at alpha = " + str(alpha) + " is:") 
        print(confusion_matrix(np.array(y_test), np.array(preds)))


    for ntree in trees:

        # Randomly split
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33)
        
        X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test), 
        y_train, y_test = pd.DataFrame(y_train), pd.DataFrame(y_test)

        # Create forest
        forest = rf.RandomForest(X_train, y_train, n_trees = ntree)
        preds = forest.predict(X_test)
        print("Confusion matrix with " + str(ntree) + " trees is: ")
        print(confusion_matrix(np.array(y_test), np.array(preds)))
 
