import pandas as pd
import numpy as np
import classification_tree as ct

class RandomForest(): 
    """ A random forest used for classification. 
    
    Instances have several attributes:
    
    x_train: Pandas dataframe of all x columns to be used for classification. All columns 
        must be of type numeric and have column names that are one word long.
    
    y_train: Pandas dataframe of one column, labelled y, which is comprised of 0's and 1's only.
        Corresponds to the respective outcomes of the observations in the x_train dataframe.
        
    folds: int, The number of cross-validation folds used for each tree in the random forest.
    
    alphas: float, List of alphas to be tested through cross-validation. For each tree, the alpha with 
        the best prediction rate becomes the alpha used to prune the tree. Note that the best
        alpha value from cross validation can be different for any particular tree.
    
    criterion: string, The method of impurity reduction used to determine impurity. Defaulted to "bayes_error",
        but may also be "cross_entropy" or "gini".
        
    n_trees: int, The number of decision trees for the random forest. Defaults to 5.
    
    n_features: int, The number of columns in the x_train Pandas dataframe to be used for each tree. 
        Defaults to the number of columns in x_train matrix if the x_train matrix only has one row,
        else the number of columns in the x_train matrix minus 1.
    
    sample_size: int, The number of rows in the x_train Pandas dataframe to be used for each tree.
    
    
    """
    def __init__(self, x_train, y_train, criterion = "bayes_error", n_trees = 5, 
                n_features = None, sample_size = None):
        """ Create random forest. """

        self.x_train = x_train   
        self.rows = self.x_train.shape[0]
        self.xcols = self.x_train.shape[1]
        self.criterion = criterion

        self.y_train = y_train
        
        if n_features is None:
            if self.xcols == 1:
                n_features = self.xcols
            else:
                n_features = self.xcols - 1
        
        if sample_size is None:
            sample_size = round(self.rows/2)
            
        self.n_trees = n_trees
        self.n_features = n_features
        self.sample_size = sample_size

        # Input assertions

        assert n_trees > 0 and type(n_trees) is int , "n_trees must be positive integer"
        assert n_features > 0 and type(n_features) is int, "n_features must be positive integer"
        assert n_features <= x_train.shape[1], "n_features must be less than or equal to number of columns in x_train"
        assert sample_size > 0 and type(sample_size) is int, "sample_size must be positive integer"
        assert sample_size <= x_train.shape[0], "sample_size must be less than or equal to number of rows in x_train"

        self.trees = [self.build_tree() for i in range(n_trees)]

    def build_tree(self):
        """Build tree, starting with parent node, by splitting variable, rows randomly.
    
        
        1. Start with parent node. 
        2. Create lhs and rhs nodes. 
        3. Use recursion so that each of the children) have children, etc.
    
        Returns
        -------
        A list of trees 
        
        Examples
        --------
        >>> self.trees = self.build_trees()
        
        See Also
        --------
        - get_data: Uses the past_split node while maintaining the x_train and y_train data as separate entities
        - find_impurity: Finds impurity of a particular cutoff point / variable pair given the criterion measure

        """
        # List of all rows/columns in X
        row_list = list(range(self.rows))
        x_col_list = list(range(self.xcols))
        
        # Randomly columns in X
        x_col_rand = np.random.choice(x_col_list, self.n_features, False) # Sample without replacement
    
        
        # Make sure that the corresponding y's chosen for the rows have 0's and 1's
        
        uniquey = 0
        while uniquey != 2:
            x_row_rand = np.random.choice(row_list, self.sample_size) # Sample with replacement
            y_train = self.y_train.iloc[x_row_rand, :]
            uniquey = len(y_train.iloc[:,0].unique())
        
        # Fit decision tree with sampled data
        x_train = self.x_train.iloc[x_row_rand, x_col_rand]
        tree = ct.DecisionTree(x_train, y_train, self.criterion)
        return tree
        
    
    def rf_predict_row(self, xi):
        
        """Take the most popular values (mode) from the trees for one x row.
    
        Parameters
        ----------
        xi: Individual row from X test dataframe.

        Returns
        -------
        Average of y value predicted by decision trees. If there is a tie, then it produces both with 
        a warning that there was a tie.
        
        """
   
        predictions = [tree.predict(xi) for tree in self.trees]
        concensus = np.mean(predictions).round()
        return concensus
    
    def predict(self, new_x):
        
        """Average the predictions of the trees for each observation. 
    
        Parameters
        ----------
        new_x: Entire X test dataframe. Must have same columns and types as x_train.

        Returns
        -------
        preds: predictions of all y values for the new x observations.

        """
        preds = [self.rf_predict_row(pd.DataFrame(new_x.iloc[i,:]).T) for i in range(new_x.shape[0])]
        return preds
