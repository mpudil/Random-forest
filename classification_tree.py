import numpy as np
import pandas as pd
import random
import warnings

# General functions to be used throughout creating classes
def gini(p):
    """Calculate gini index for given p.

    Parameters
    ----------
    p: numeric, a probability.

    Returns
    -------
    p*(1-p)

    Examples
    --------
    gini(0.4)

    """
    # Assert p in range 0-1
    assert 0 <= p <= 1, "p must be in range 0-1"
    gini_index = p*(1-p)
    return gini_index
    
    
def cross_entropy(p):
    """Calculate cross-entropy.

    Parameters
    ----------
    p: numeric, a probability.

    Returns
    -------
    -p*log(p) - (1-p)*log(1-p)

    Examples
    --------
    cross_entropy(0.4)

    """
    assert 0 < p < 1, "p must be in range 0-1"
    ce =  -p * np.log(p) - (1-p)*np.log(1-p)
    return ce



def bayes_error(p):
    """Calculate cross-entropy for given p.

    Parameters
    ----------
    p: numeric, a probability 

    Returns
    -------
    min(p, 1-p)

    Examples
    --------
    bayes_error(0.4)

    """
    assert 0 <= p <= 1, "p must be in range 0-1"
    be = min(p,1-p)
    return be

def chunkIt(seq, num):
    """Split an array into k pieces of approximately equal length.
    
    Parameters
    ----------
    seq: an array, list, or range.
    num: number of approximately equal pieces desired.

    Returns
    -------
    chunked_list: an array that has been separated into n=num pieces.

    Examples
    --------
    chunkIt(range(1,100), num=5)
    """
    assert num > 0 and type(num) is int, "num must be an integer greater than 0"
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out 


def subset(S, xj, sign, df):
    """Return subset of data for a particular cutoff. 
    Parameters
    ----------
    S: cutoff point to be subsetted on
    xj: variable to be used to subset data
    sign: Currently only accepts ">" and "<="; used for relationship between 
    subsetted and full df
    
    Returns
    -------
    x_subset, y_subset: subset based off of just one comparison

    Examples
    --------
    self.get_data()

    """
    assert sign == "<=" or sign == ">", "Sign must be <= or >"
    if sign == "<=":
        return(df.loc[df[xj] <= S])
    if sign == ">":
        return(df.loc[df[xj] > S])


def get_data_in_region(full_x_df, full_y_df, path):
    """Return x and y dataframes where x_train matches particular path.

    Parameters
    ----------
    full_x_df: full x dataframe.
    full_y_df: full y dataframe.
    path: a list of paths e.g. ['var1 < 3', 'var2 > 7', 'var1 <= 1.4'].

    Returns
    -------
    X_train dataframe subsetted to include path.
    Y_train dataframe subsetted to include same path.

    Examples
    --------
    get_data_in_region(x, y, ['var1 < 3', 'var2 > 7', 'var1 <= 1.4'])

    """
    x_subset = full_x_df.copy()
    for p in path:
        x_subset = eval("x_subset[x_subset." + p + "]")

    y_subset = full_y_df.loc[x_subset.index.values,:]

    return x_subset, y_subset

def misclass_cost(y, nrow):
    """Determine the misclassification cost of a particular node.

    Parameters
    ----------
    y: dataframe or matrix of all y in a particular (or potential) node
    nrow: total number of rows in entire tree

    Returns
    -------
    misclass_cost: misclassification cost associated with the node

    """ 
    correct_y = np.mean(y).round()
    y_length = y.shape[0]
    n_misclass = np.sum(y != correct_y)[0] / y_length
    frac_all = y_length / nrow
    misclass_cost = n_misclass * frac_all
    return misclass_cost



class Node:
    """ A node of a classification tree.
    
    Instances have several attributes:
    
    x_train: a dataframe of all x columns to be used for classification. All columns 
        must be of type numeric and have column names that are one word long.
    
    y_train: a dataframe of one column, labelled y, which is comprised of 0's and 1's only.
        Corresponds to the respective outcomes of the observations in the x_train dataframe.
    
    criterion: The method of impurity reduction used to determine impurity. Defaulted to "Bayes",
        but may also be "Cross-entropy" or "Gini" or a different impurity function written in 
        terms of p/
        
    path: a list of lists denoting the path taken to arrive at a particular node. 
        E.g. [["var1 < 3"], ["v2 > 8"]]. Note that this functionality is primarily used
        for in the SQLNode class rather than the Node class.
        
    nrow: Number of rows in the dataframe (note that this can be automatically determined).
    
    
    """
        
        
    def __init__(self, x_train, y_train, criterion = "bayes_error", path = [], nrow = None, S=None, 
                xj=None, lhs = None, rhs = None):
        """Initialize node, calculate nrow, assert valid inputs."""
        #if nrow is None:
            
        
            
        self.nrow = nrow
        # Matrix of explanatory variables for training model
        self.x_train = x_train
    
        # Matrix of dependent variable for training model
        self.y_train = y_train
        
        # Assert inputs are correct (only need to do this once at beginning)
        self.path = path
        if len(self.path) == 0:
            self.nrow = x_train.shape[0]
            
            assert self.x_train.shape[0] == self.y_train.shape[0], "X and Y must have same number of rows"
            assert self.y_train.shape[1] == 1, "Y should have 1 column"
            assert sum(np.sum(self.x_train.isnull())) == 0, "X dataframe must have no missing data"
            assert sum(np.sum(self.y_train.isnull())) == 0, "Y dataframe must have no missing data"
            
            for col in self.x_train.columns:
                if len(self.x_train[col].unique()) == 1:
                    warnings.warn("Column '" + str(col) + "' has been ignored as it contains only one unique value")

            assert len(self.y_train.iloc[:,0].unique()) == 2, "Y dataframe must have 0's and 1's"
        
        self.S = S
        self.xj = xj
        
        
        # Stores indexes of the subset of the data that the node is working with
        self.x_train.index = range(self.x_train.shape[0])
        self.y_train.index = range(self.y_train.shape[0])
        
        #self.idxs = range(len(x_train)) 
        self.row_count = x_train.shape[0] # Number of rows in dataset
        self.col_x_train = x_train.shape[1] # Number of columns of x_train
        
        self.criterion = criterion        
        # Create children iteratively
        
        self.make_children()
        
      

    def potential_splits(self,potential_xj): 
        """Return potential splitting points for a particular column of x_train: potential_xj.

        Require that there are multiple values in potential_xj (not all same number). 
        
        Then split potential_xj into several regions by determining quartile (if continuous) 
        or else returning the unique discrete values.

        Parameters
        ----------
        potential_xj: Column of x_train that contains the variable we are considering.
        using as the xj for the cutoff.

        Returns
        -------
        potential_S_list: list of potential cutoff values to try based off quartile.

        """
        potential_S_list = list(set(self.x_train[potential_xj]))
        # Set ensures no duplicates
            
        return potential_S_list
        

    
    def impurity_reduction(self, potential_xj, potential_S):
        """Return impurity reduction deltaI(S,A) of a particular cutoff point / variable pair 
        given the criterion measure.

        Parameters
        ----------
        self : all data from init

        potential_xj: Column of x-train that we would like to test as the potential cutoff variable.
        potential_S: Potential cutoff point we would like to test as the potential cutoff point (element
                    of potential_S_list).

        Returns
        -------
        change_impurity: change in impurity. 


        """
        
        left_rows = list(self.x_train[self.x_train[potential_xj] <= potential_S].index.values)
        left_y_temp = self.y_train.loc[left_rows, :]
        right_y_temp = self.y_train.drop(left_rows, axis=0)
        
        if left_y_temp.shape[0] == 0 or right_y_temp.shape[0] == 0:
            return 0
        
        else:
        
            # prob_one_X is number of observations in the node that are 1 / total number of observations
            # Note this is equivalent to the mean of the respective y column since y can only either be 0 or 1.
            prob_one_all = float(np.mean(self.y_train))
            prob_one_left = float(np.mean(left_y_temp))
            prob_one_right = float(np.mean(right_y_temp))

            p_L = left_y_temp.shape[0] / self.y_train.shape[0]
            p_R = right_y_temp.shape[0] / self.y_train.shape[0]

            I_A = eval(str(self.criterion) + "(" + str(prob_one_all) + ")")
            I_L = eval(str(self.criterion) + "(" + str(prob_one_left) + ")")
            I_R = eval(str(self.criterion) + "(" + str(prob_one_right) + ")")

            change_impurity = I_A - p_L*I_L - p_R*I_R

            return change_impurity


    def best_split(self): 
        """Find best cutoff point S for a particular column of x_train: xj, 
        i.e. the pair that reduces impurity the most.
  
        For each variable, loop through a list of possible cutoffs suggested by 
        potential_splits function to find the cutoffs that would decrease the impurity 
        the most if the cutoff had to be that variable. 
        Then compare the impurities across the variables and chooses the cutoff/variable 
        mix with the lowest impurity. If tie, use first of the pairs calculated.

        Parameters
        ----------
        self: see __init__

        Returns
        -------
        S: cutoff to use for split (numeric)
        xj: variable to use for split (string)
        
        
        Examples
        --------
        self.S, self.xj = self.best_split()
        

        """
        
        impurity, best_S, best_xj = 0, None, None
        
        # Iterate to determine best cutoff split and variable.
        for xj in list(self.x_train.columns):
            for S in self.potential_splits(xj):
                ir = float(self.impurity_reduction(xj, S))
                if ir > impurity:
                    impurity, best_S, best_xj = ir, S, xj
                else: 
                    pass
        return best_S, best_xj
        

    def make_children(self): 
        """Create lhs and rhs for node based off cutoff value S and x_train column xj.

        Require that each child would have data and that the node is not a leaf.
        Require that all_same_responses is False.
        Use row index so that X matrix and Y matrix line up.
        
        Returns
        -------
        self (with additional attributes lhs and rhs)
        
        self.lhs: Node class with where rows of the x_train and y_train data correspond to values that 
        are less than S on the xj cutoff variable. Appended a row to list of splits.

        self.rhs: Node class with where rows of the x_train and y_train data correspond to values that 
        are greater than or equal to S on the xj cutoff variable
        """
        
        S, xj = self.best_split()
        
        # May want to put this in __init__
        if S is None and xj is None:
            self.lhs = None
            self.rhs = None
            return 
        
        
        left_rows = list(self.x_train[self.x_train[xj] <= S].index.values)
        left_x = self.x_train.loc[left_rows, :]
        left_y = self.y_train.loc[left_rows, :]
 
        right_rows = list(self.x_train[self.x_train[xj] > S].index.values)
        right_x = self.x_train.loc[right_rows, :]
        right_y = self.y_train.loc[right_rows, :]
        
        if left_y.shape[0] == 0 or right_y.shape[0] == 0:
            return 
        
        else:
        # Make lhs and rhs nodes (children)    
            self.lhs = Node(left_x, left_y, self.criterion, self.path + [xj,  ' <= ', str(S)], self.nrow, S, xj)
            self.rhs = Node(right_x, right_y, self.criterion, self.path + [xj, ' > ', str(S)], self.nrow, S, xj)
            
            return 
        
        
    @property 
    def is_leaf(self): 
        """ Determine if self node is a leaf (has no children).
        
        Returns
        -------
        Boolean: True if has no children, i.e. is leaf, False otherwise.
        """
        return self.rhs is None and self.lhs is None
    
    def predict_row(self, xi):
        """Predict y value for new, single row in X matrix.

        Parameters
        ----------
        xi: A row of the new x dataframe to be classified.

        Returns
        -------
        prediction of y value.

        """
        S, xj = self.best_split()
        if self.is_leaf:
            return round(np.mean(self.y_train))  # mean of y values
        else:
            node = self.lhs if (float(xi[xj]) <= S) else self.rhs 
            return node.predict_row(xi)

    def predict(self, x):
        """Predict entire Y matrix for all X's.

        Parameters
        ----------
        self : all data from init 

        Returns
        -------
        prediction of y value
        
        Examples
        --------
        tree.predict(X_df)
        
        See Also
        --------
        - predict_row

        """
        if x.shape[0] == 1:
            prediction = [self.predict_row(x)]
        else:
            prediction = [self.predict_row(pd.DataFrame(x.iloc[i,:]).T) for i in range(x.shape[0])]
            
        return prediction

    def diff_misclass_cost(self):
        """Determine difference between parent and children misclassification cost. 

        Calculates r(t) and p(t) 
        Multiplies r(t) and p(t) for each child (rhs and lhs)
        Adds above together to find R(t)
        R(t) - R(T_t)

        Parameters
        ----------
        self : all data from init 

        Returns
        -------
        cost: misclassification cost R(t) associated with the tree

        """
        # Calculating Parent Misclass Cost, i.e. R(t)
        misclass_cost_parent = misclass_cost(self.y_train, self.nrow)

        misclass_cost, 
        # Left misclasification cost
        misclass_cost_left = misclass_cost(self.lhs.y_train, self.nrow)
       
        # Right misclass cost
        misclass_cost_right = misclass_cost(self.rhs.y_train, self.nrow)
        
        # Total Cost g(t)

        # Since nleaves will always be 2, and 2-1 = 1, then 
        # g(t) = (misclass_cost_parent - misclass_cost_children) / abs(leaves)-1 = 
        # misclass_cost_parent - misclass_cost_children
        misclass_diff = misclass_cost_parent - sum([misclass_cost_left, misclass_cost_right])

        return misclass_diff

    
    
    def prune(self, alpha): 
        """Prune tree by comparing alpha with misclassification cost. """
        if len(self.path) > 0:
            if not self.lhs.is_leaf:
                self.lhs.prune(alpha)

            if not self.rhs.is_leaf:
                self.rhs.prune(alpha)

            if self.lhs.is_leaf and self.rhs.is_leaf:
                if self.diff_misclass_cost() <= alpha:
                    self.lhs, self.rhs = None, None
                else:
                    pass
        else:
            return

class DecisionTree():
    """ A decision tree: a hierarchical structure of nodes used for classification.
    
    Instances have several attributes:
    
    x_train: a dataframe of all x columns to be used for classification. All columns 
        must be of type numeric and have column names that are one word long.
    
    y_train: a dataframe of one column, labelled y, which is comprised of 0's and 1's only.
        Corresponds to the respective outcomes of the observations in the x_train dataframe.
    
    criterion: The method of impurity reduction used to determine impurity. Defaulted to "gini",
        but may also be "cross_entropy" or "bayes" or a different impurity function written as
        a stringed function of p in Python syntax, e.g. "min(p**2, (1-p) )"
        
 
    
    """
    
    def __init__(self, x_train, y_train, criterion = "bayes_error"):
        """ Initialize decision tree based off of user's x_train, y_train, and criterion.
        Perform this by calling the Node function, which iteratively makes children.
        Note that user should afterwards call self.prune with desired alpha list (or default alpha)
        to avoid overfitting.
        """
        assert type(x_train) is pd.core.frame.DataFrame, "x_train must be Pandas dataframe"
        assert type(y_train) is pd.core.frame.DataFrame, "y_train must be Pandas dataframe"
        assert criterion == "bayes_error" or criterion=="gini" or criterion=="cross_entropy"
        assert x_train.shape[0] > 0 and x_train.shape[1] > 0, "x_train must have positive number of rows and columns"
        assert x_train.shape[0] == y_train.shape[0], "x_train and y_train must have same number of rows"
        assert y_train.shape[1] == 1, "y_train must have only one column"
        assert np.array(set(y_train.iloc[:,0])) == {0,1}, "y_train must be composed of 0's and 1's"
        

        self.nrow = x_train.shape[0] 
        self.x_train = x_train
        self.y_train = y_train
        self.criterion = criterion
        
        self.tnode = Node(self.x_train, self.y_train, criterion=self.criterion)
        self.nrows_leaves = 0

    
    def predict(self, new_x):
        """ Predict y_values for given new_x dataframe.

        Parameters
        ----------
        new_x: nxk Pandas dataframe of X's with same columns as x_train dataframe. 

        Returns
        -------
        Array of predictions for each observation.
        
        """
        return self.tnode.predict(new_x)


    def cross_validate(self, folds=5, alphas=[0.1, 0.5, 0.7]):
        """Perform k-fold cross validation and return best alpha from alphas list.
        
        Asserts that each alpha in alphas >= 0.
        Splits the data in self into k (default 5) folds. Determines test set error for each 
        value of alpha in the alphas list. Returns the alpha that leads to the smallest error.

        Parameters
        ----------
        self : all data from init.
        folds: number of folds (k in k-folds cross validation).
        alphas: list of alphas to be tested.

        Returns
        -------
        alpha_star: element in alphas that led to smallest test set error.

        """
        
        #np.random.seed(100)
        rows = self.x_train.index.values
        random.shuffle(rows) 
        kfoldrows = chunkIt(rows, folds)
        alpha_star = 0
        lowest_error = float('inf')
        
        for alpha in alphas:
            assert alpha >= 0, "All alphas must be at least 0"
            errors = [None]*folds

            for fold in range(folds):  

                # Create training and testing datasets
                unique_train_y = 0

                # Make sure there are 0's and 1's in the y train
                while unique_train_y != 2:
                    test_rows = kfoldrows[fold]
                    y_train_cv = self.y_train.drop(list(test_rows), axis=0)
                    unique_train_y = len(set(y_train_cv.iloc[:,0]))

                y_test_cv = self.y_train.iloc[test_rows]
                x_test_cv = self.x_train.iloc[test_rows]
                x_train_cv = self.x_train.drop(list(test_rows), axis=0)
                
    
                # Fit and predict
                fit = DecisionTree(x_train_cv, y_train_cv)
                preds = fit.predict(x_test_cv)
                wronglist = np.subtract(np.transpose(np.array(y_test_cv.values)), preds) 
                errors[fold] = np.mean([abs(wrong) for wrong in wronglist])
            
            # Capture results, update alpha* accordingly
            mean_error = np.mean(errors)
            if mean_error < lowest_error:
                lowest_error = mean_error
                alpha_star = alpha
           
        
        return alpha_star
    

    def prune(self, folds = 5, alphas = [0.1, 0.5, 0.7], cross_validate = True): 
        """Prune tree using cross-validation from user-provided alphas list. Note if 
        specific alpha is known by user, then setting the alphas list equal to a 
        one-element array alpha, e.g. alphas = [0.1], will ensure that that alpha is chosen.
        
        Start from the second-to-last layer from bottom of tree using self.xhs.is_leaf() 
        and sees if that subtree should be pruned. If it should, look at its parent and 
        see if it should be prune, etc. until should_prune() returns false, suggesting in
        which case prune its child(ren), but not itself.
        
        Parameters
        ----------
        self : all data from init, i.e. a tree.

        Returns
        -------
        Pruned tree
        
        Examples
        --------
        tree.prune(folds = 2, alphas = [0.1, 0.4])
    
        """
        if self.tnode.is_leaf is True:
            return
        
        if cross_validate is True:
            alpha = self.cross_validate(folds, alphas)
        else:
            alpha = alphas[0]
        
        if alpha == 0:
            return
        else:
            self.tnode.prune(alpha) 
        return
    
    @property
    def is_valid(self):
        """ Determines if a tree is valid.
        
        A tree is valid when the following are true:
        1. No nodes are empty
        2. If a node is split on xj, then all data points in node.lhs should be sure that xj <= S
            and all data points in node.rhs should be where xj > S
        3. Applying the get_data function yields same dataset (esp. length of dataset) as 
            applying to root node. 
            
        Returns
        -------
        Boolean: true if tree is valid, false otherwise.
        
        Examples
        --------
        tree.is_valid
    
        """
        return self.no_nodes_empty(self.tnode) and self.correct_splits(self.tnode) and self.correct_nleaves(self.tnode)
    

    def no_nodes_empty(self, root):
        """ Check that no nodes are empty, i.e. that they all contain data.
        
        Parameters:
        ----------
        root: root node of tree.
        
        Returns:
        --------
        Boolean: True is no nodes empty, False otherwise.
        
        """
        result = True
        if root:
            if root.y_train.empty:
                return False
            self.no_nodes_empty(root.lhs)
            self.no_nodes_empty(root.rhs)
            
        return result
                
        

    def correct_splits(self, root):
        """ Check that if node split on xj, then all data points in lhs is where xj <= S 
        and all points in rhs has xj > S.
        
        Parameters:
        ----------
        root: root node of tree.
        
        Returns:
        --------
        Boolean: True if all splits are correctly sorted, False otherwise.
        
        """
        result = True
        
        if root.is_leaf:
            if len(root.path) > 0:
                S = root.S
                xj = root.xj
                if all(root.x_train[xj] <= S) is False or all(root.x_train[xj] > S) is False:
                    result = False
                
                self.correct_splits(root.lhs)
                self.correct_splits(root.rhs)
    
        return result
        
            
    def correct_nleaves(self, root, nrows_leaves = 0):
        """ Determine if the number of rows of y in each of the leaves 
            sums to total number of rows.
        
        Parameters:
        ----------
        root: root node of tree
        
        Returns:
        --------
        Boolean: True if all splits are correctly sorted, False otherwise
        
        """

        if root:
            if root.is_leaf:
                self.nrows_leaves += root.y_train.shape[0]
            
            self.correct_nleaves(root.lhs, nrows_leaves)
            self.correct_nleaves(root.rhs, nrows_leaves)
        
        
        return self.tnode.nrow == self.nrows_leaves