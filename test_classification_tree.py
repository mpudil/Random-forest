import pytest
import numpy as np
import pandas as pd
import warnings
import classification_tree as ct
import credentials as cd
import csv

# R1 tests

@pytest.fixture
def r1_tree():
    # if x <= -2, then y = 1, if -2 <= x < 9, then y = 0, else y = 1
    x_data = [[-10], [-5], [-2], [0], [2], [6], [9], [12], [20], [100]]
    x_df = pd.DataFrame(x_data, columns=['var1'])
    y_data = [1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    y_df = pd.DataFrame(y_data, columns=['y'])
    x_test_data = [[-10], [5], [8], [12], [50]]
    x_test_df = pd.DataFrame(x_test_data, columns=['var1'])
    t1 = ct.DecisionTree(x_df, y_df)
    return t1, x_test_df

def test_valid_r1_tree(r1_tree):
    t1, _ = r1_tree
    assert t1.is_valid


def test_children_r1(r1_tree):
    t1, _ = r1_tree
    assert t1.tnode.lhs.x_train.shape == (7, 1)
    assert t1.tnode.lhs.y_train.shape == (7, 1)
    assert t1.tnode.rhs.x_train.shape == (3, 1)
    assert t1.tnode.rhs.y_train.shape == (3, 1)

def test_paths_r1(r1_tree):
    t1, _ = r1_tree
    assert t1.tnode.lhs.path == ['var1', ' <= ', '9']
    assert t1.tnode.lhs.rhs.path == t1.tnode.lhs.path + ['var1', ' > ', '-2']
    
    
def test_predict_r1(r1_tree):
    t1, x_test_df = r1_tree
    assert all(np.array(t1.predict(x_test_df)) == np.array([[1.], [0.], [0.], [1.],[1.]]))

def test_diff_misclass_cost_r1(r1_tree):
    t1, _ = r1_tree
    assert round(t1.tnode.diff_misclass_cost(), 5) == 0.1



# R2 tests
@pytest.fixture
def r2_tree():
    # Split into quadrants: y = 0 if x in Quad. 1 or 4, and 1 otherwise
    x_data = [[1,5], [4,2], [1,1], [2,4], [3,1], [-1, 6], [-2,9], [-5, 1],
              [-7, 1], [-6, 4], [-0.1, 0.1], [-4, -5], [-1, -2], [-0.2, -5],
              [3, -4], [5, -1]]
    x_df = pd.DataFrame(x_data, columns=['var1', 'var2'])
    y_data = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
    y_df = pd.DataFrame(y_data, columns=['y'])
    tree = ct.DecisionTree(x_df, y_df)
    return tree, x_df, y_df


def test_valid_r2_tree(r2_tree):
    t2, _, _ = r2_tree
    assert t2.is_valid

# Basic splitting test 
def test_splits_basic(r2_tree):
    t2, _, _ = r2_tree
    assert t2.tnode.lhs.path == ['var1', ' <= ', '-0.1']
    assert t2.tnode.rhs.path == ['var1', ' > ', '-0.1']
    
def test_children_r2(r2_tree):   
    t2, _, _ = r2_tree
    assert t2.tnode.lhs.x_train.shape == (9, 2)
    assert t2.tnode.lhs.y_train.shape == (9, 1)
    assert t2.tnode.rhs.x_train.shape == (7, 2)
    assert t2.tnode.rhs.y_train.shape == (7, 1)
    
def test_predict_r2(r2_tree):
    t2, _, _ = r2_tree
    x_test_data = [[-1, 7], [-1, -5], [-3, -4], [4, -2]]
    x_test_df = pd.DataFrame(x_test_data, columns = ['var1', 'var2'])
    assert all(np.array(t2.predict(x_test_df)) == np.array([[1.], [0.], [0.], [1.]]))

def test_predict_trained_100(r2_tree):
    # The unpruned tree should get a 100% accuracy for training dataset
    t2, x_df, y_df = r2_tree
    preds_train = pd.DataFrame(t2.predict(x_df), columns = ['y'])
    preds_train.y = preds_train.astype('int')
    assert all(preds_train == y_df)


@pytest.fixture
def r3_tree():
    # y = 1 if x1 <= 0 and x2 <= 0 or x1 > 0 and x3 <= 0. Otherwise y = 0
    x_data = [[-5, -4, -1], [-5, -1, -3], [2, -1, 17], [2.5, 5, -10], [2, 2, 2], 
        [-6, 1, 8], [4, -5, 0], [5, 0, 0.5], [16, 4, -5], [100, 3, 3], [4, -10, 17], 
        [8, -1, 5], [-3, 5, 6], [-0.1, -0.1, 0.1], [10, -0.1, -4]]
    x_df = pd.DataFrame(x_data, columns=['var1', 'var2', 'var3'])
    y_data = [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]
    y_df = pd.DataFrame(y_data, columns=['y'])

    x_test_data = [[-1, -1, 5], [1, 2, -1], [6, 0, 0], [5, -1, 6]]
    x_test_df = pd.DataFrame(x_test_data, columns=['var1', 'var2', 'var3'])

    t3=ct.DecisionTree(x_df, y_df)
    return t3, x_test_df, y_df
 

def test_children_r3(r3_tree): 
    t3, _, _ = r3_tree
    assert t3.tnode.lhs.x_train.shape == (5, 3)
    assert t3.tnode.lhs.y_train.shape == (5, 1)
    assert t3.tnode.rhs.x_train.shape == (10, 3)
    assert t3.tnode.rhs.y_train.shape == (10, 1)

def test_predict_r3(r3_tree):
    t3, x_test_df, _ = r3_tree
    assert all(np.array(t3.predict(x_test_df)) == np.array([[1.], [1.], [0.], [0.]]))

def test_misclass_cost_r2(r3_tree):
    t3, _, _ = r3_tree
    assert round(t3.tnode.diff_misclass_cost(), 3) == 0.333

# Question: What if responses are all the same value i.e. 
# all 1's or all 0's?
# Answer: Nodes should not be made (and therefore no predictions, 
# fitting, pruning, etc.)

def test_same_response():
    x_data = [[0, 2, 4], [1, 2, 3], [4, 3, 1]]
    x_df = pd.DataFrame(x_data, columns=['Gender', 'Age', 'Income'])
    y_data = [1, 1, 1]
    y_df = pd.DataFrame(y_data, columns=['Pass'])

    try:
        ct.Node(x_df, y_df)
    except AssertionError:
        "Y dataframe must have 0's and 1's"
        
# Question: How will you deal with the missing values e.g. delete them, 
# impute them (how?) 
# Answer: Stop and assert error that there are missing values in the dataframe

def test_missing_values():
    x_data = [[0, 2, 4], [1, None, 3], [4, 2, 1]]
    x_df = pd.DataFrame(x_data, columns=['Gender', 'Age', 'Income'])
    y_data = [1, 0, 1]
    y_df = pd.DataFrame(y_data, columns=['y'])

    try:
        ct.Node(x_df, y_df)
    except AssertionError:
        "X dataframe must have no missing data"
        
        
# Question: What if a split feature is all the same value?
# Answer: The column should be deleted in __init__ before prediction
# Prediction, fitting, etc. should still work, but there should be a warning that 
# the column has been ignored

def test_same_value_x():
    x_df = pd.DataFrame([[1,2], [1,4], [1,6], [1, 7]], columns = ['var1', 'var2'])
    y_df = pd.DataFrame([0,0,0,1], columns = ['y'])
    with pytest.warns(UserWarning):
        ct.Node(x_df, y_df)       
        
        
def import_format_tests_x():
    x_data = [[0, "the"], [0, 1], [2, 3]]
    x_df = pd.DataFrame(x_data, columns=['var1', 'var2'])
    y_data = [0, 1, 0]
    y_df = pd.DataFrame(y_data, columns=['y'])
    with pytest.raises(TypeError):
        ct.Node(x_df, y_df)


def test_gini():
    assert [ct.gini(0), ct.gini(0.5), ct.gini(1)] == [0, 0.25, 0]
    
def test_cross_entropy():
    assert ct.cross_entropy(0.5).round(2) == 0.69

def test_bayes_error():
    assert ct.bayes_error(0.5) == 0.5
    
def test_chunkIt():
    assert ct.chunkIt(list(range(10)), 4) == [[0, 1], [2, 3, 4], 
        [5, 6], [7, 8, 9]]
    assert ct.chunkIt([1, 1, 1, 1], 3) == [[1], [1], [1, 1]]


def test_potential_splits():
    x_train = [[1, 2, 3], [4, 5, 6], [6, 7, 8], [1, 1, 2], 
        [-1, 4, 5], [0, 0, 1]]
    y_train = [0, 1, 1, 0, 1, 1]
    x_df = pd.DataFrame(x_train, columns = ['v1', 'v2', 'v3'])
    y_df = pd.DataFrame(y_train, columns=['y'])
    n = ct.Node(x_df, y_df)
    assert n.potential_splits('v1') == [0, 1, 4, 6, -1]
    
    
    
def test_impurity_reduction():
    x_train = [[1, 2, 3], [4, 5, 6], [6, 7, 8], [1, 1, 2], [-1, 4, 5], 
        [0, 0, 1]]
    y_train = [0, 1, 1, 0, 1, 1]
    x_df = pd.DataFrame(x_train, columns = ['v1', 'v2', 'v3'])
    y_df = pd.DataFrame(y_train, columns=['y'])
    n = ct.Node(x_df, y_df)
    assert float(round(n.impurity_reduction('v2', 3), 2)) == 0.17

    
def test_best_split():
    x_train = [[1,2,3], [4,5,6], [6,7,8], [1,1,2], [-1,4,5], [0,0,1]]
    y_train = [0, 1, 1, 0, 1, 1]
    x_df = pd.DataFrame(x_train, columns = ['v1', 'v2', 'v3'])
    y_df = pd.DataFrame(y_train, columns=['y'])
    n = ct.Node(x_df, y_df)
    assert n.best_split() == (2, 'v2')
       

# DecisionTrees with random data are valid
def test_random_valid():
    x_df = pd.DataFrame(np.random.randn(50, 5), columns=range(5))
    y_df = pd.DataFrame(np.random.random((50,1)), columns=['y']).round()
    dt = ct.DecisionTree(x_df, y_df)
    assert dt.is_valid
 
  

# If two splits reduce the impurity by the same amount, the first split is chosen

def test_two_best_splits():
    # Note that splitting on x1 @ 0 and x2 @ 0 both split the data perfectly
    x_train = [[3.1,0.1], [4,1], [7, 2], [6, 3], [0, 0], [-1, 3], [-4.1, 2.8], [-10, -10]]
    y_train = [1, 1, 1, 1, 0, 0, 0, 0]
    x_df = pd.DataFrame(x_train, columns = ['v1', 'v2'])
    y_df = pd.DataFrame(y_train, columns=['y'])
    dtree = ct.DecisionTree(x_df, y_df)
    assert dtree.tnode.lhs.path == ['v1', ' <= ', '0.0']