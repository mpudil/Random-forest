import random
import time
import cProfile
import numpy as np
import classification_tree as ct
import pandas as pd
import random_forest as rf

cols = [2, 4, 6, 8, 10, 15]
rows = [25, 50, 75, 100, 200]

time_dt = []
time_prune = []
time_dt_predict = []
time_rf = []
time_rf_predict = []

def time_tree(method):
    t0 = time.time()
    result = eval(method)
    t1 = time.time()
    time_taken = t1 - t0
    return time_taken, result 

for row in rows:
    for col in cols:
        for i in range(3):
            
            # Make train and test datasets
            x_df = pd.DataFrame(np.random.randn(row, col), columns=range(col))
            y_df = pd.DataFrame(np.random.randint(0, 2, size=(row,1)), columns=['y'])
            x_test = pd.DataFrame(np.random.randn(row, col), columns=range(col))
            
            # Time to build decision tree
            total, tree = time_tree('ct.DecisionTree(x_df, y_df)')
            time_dt.append(["CreateDT", row, col, total])
            print("Tree with " + str(row) + " rows and " + str(col) + " columns created in " +  str(total) + " seconds.")

            # Time to prune (default alphas)
            total, _ = time_tree('tree.prune()')
            time_prune.append(["PruneDT", row, col, total])
            print("Tree with " + str(row) + " rows and " + str(col) + " columns pruned in " + str(total) + " seconds.")
         
            # Time to predict (Decision Tree)
            
            total, _ = time_tree('tree.predict(x_test)')
            time_dt_predict.append(["PredictDT", row, col, total])
            print("Tree with " + str(row) + " rows and " + str(col) + " columns made prediction in " + str(total) + " seconds.")
            

            # Time to build random forest
            
            total, forest = time_tree('rf.RandomForest(x_df, y_df)')
            time_rf.append(["BuildRF", row, col, total])
            print("Forest with " + str(row) + " rows and " + str(col) + " columns created in " + str(total) + " seconds.")
            
            # Time to predict (Random Forest)
            total, _ = time_tree('forest.predict(x_test)')
            time_rf_predict.append(["PredictRF", row, col, total])
            print("Forest with " + str(row) + " rows and " + str(col) + " columns made predictions in " + str(total) + " seconds.")


def time_buildtree():
     return ct.DecisionTree(x_df,y_df)

cProfile.run('time_buildtree()')