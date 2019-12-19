import numpy as np
import pandas as pd
import pytest
import random_forest as rf
import random

random.seed(12)



@pytest.fixture
def forest_data():
    x_data = [[1,5], [4,2], [1,1], [2,4], [3,1], [-1, 6], [-2,9], [-5, 1],
              [-7, 1], [-6, 4], [-0.1, 0.1], [-4, -5], [-1, -2], [-0.2, -5],
              [3, -4], [5, -1]]
    x_df = pd.DataFrame(x_data, columns=['var1', 'var2'])
    y_data = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
    y_df = pd.DataFrame(y_data, columns=['y'])
    f = rf.RandomForest(x_df, y_df)
    return f


def test_forest_valid_trees(forest_data):
    f = forest_data
    assert len(f.trees) == 5 
    for t in f.trees:
        # all trees are given ceiling(n/2) rows and (k-1) columns by default
        assert t.tnode.x_train.shape == (8,1) 
        assert t.is_valid


# Random Forest predicts training data with high accuracy
def test_train_forest_accuracy():
    x_df = pd.DataFrame(np.random.randn(20, 3), columns=range(3))
    y_df = pd.DataFrame(np.random.randint(0, 2, size=(20,1)), columns=['y'])
    forest = rf.RandomForest(x_df, y_df)
    preds = forest.predict(x_df)
    preds_df = pd.DataFrame([int(p) for p in preds], columns=['y'])
    # Forest should easily be able to get 70% accuracy
    assert sum(y_df.y == preds_df.y) > 0.7 

# Edge Cases:
# Test that RandomForest works with only one feature

def test_only_one_feature_forest():
    x_df = pd.DataFrame(np.random.randn(20, 1), columns=['var1'])
    y_df = pd.DataFrame(np.random.randint(0, 2, size=(20,1)), columns=['y'])
    x_test = pd.DataFrame(np.random.randn(10, 1), columns=['var1'])
    forest = rf.RandomForest(x_df, y_df)
    preds = forest.predict(x_test)
    preds_df = pd.DataFrame([int(p) for p in preds], columns=['y'])
    assert preds_df.shape == (10, 1)


 

 # Test that RandomForest works with only one tree
@pytest.fixture
def more_forest_data():
    x_df = pd.DataFrame(np.random.randn(20, 2), columns=['var1', 'var2'])
    y_df = pd.DataFrame(np.random.randint(0, 2, size=(20,1)), columns=['y'])
    x_test = pd.DataFrame(np.random.randn(10, 2), columns=['var1', 'var2'])
    return x_df, y_df, x_test


def test_one_tree(more_forest_data):
    x_df, y_df, x_test = more_forest_data
    forest = rf.RandomForest(x_df, y_df, n_trees = 1)
    preds = forest.predict(x_test)
    preds_df = pd.DataFrame([int(p) for p in preds], columns=['y'])
    assert preds_df.shape == (10, 1)


 # Test that it's ok if the number of predictions is greater than the training dataset
def test_many_preds(more_forest_data):
    x_df, y_df, _ = more_forest_data
    x_test = pd.DataFrame(np.random.randn(50, 2), columns=['var1', 'var2'])
    forest = rf.RandomForest(x_df, y_df, n_trees = 1)
    preds = forest.predict(x_test)
    preds_df = pd.DataFrame([int(p) for p in preds], columns=['y'])
    assert preds_df.shape == (50, 1)


# Test that RandomForest works with very small sample sizes
def test_few_rows_forest():
    x_df = pd.DataFrame(np.random.randn(4, 1), columns=['var1'])
    y_df = pd.DataFrame(np.random.randint(0, 2, size=(4,1)), columns=['y'])
    x_test = pd.DataFrame(np.random.randn(3, 1), columns=['var1'])
    forest = rf.RandomForest(x_df, y_df)
    preds = forest.predict(x_test)
    preds_df = pd.DataFrame([int(p) for p in preds], columns=['y'])
    assert preds_df.shape == (3, 1)






