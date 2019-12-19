# Python and SQL Classification Trees and Random Forests

Python implementation of classification tree and random forest as described [here](https://github.com/36-750/problem-bank/blob/master/All/classification-tree.pdf). With user-provided credentials, it also provides classification trees, with 
prediction, for data stored in SQL tables.

The overall organization of modules, files, and folders is as follows:

The [Empirical](https://github.com/36-750/assignments-mpudil/tree/classification-tree-3/classification-tree/Empirical) folder includes an example of real-world data used to analyze the predictive power of the 
Decision trees and random forests under various circumstances. The empirical testing was performed on the file [Acceptance.csv](hhttps://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/Empirical/acceptance.csv), which is publically-available data on acceptance to Berkeley acceptance. It is used to test prediction on a real-world dataset for the decision trees and random forest (with default cross-validation and pruning values). 

The [Resources](https://github.com/36-750/assignments-mpudil/tree/classification-tree-3/classification-tree/Resources) folder includes the SQL functions necessary to implement before creating a SQL-based tree. It also includes the paper from Breiman that details more explicitly the functionality of Random Forests in general. 

The [Timing_Profiling](https://github.com/36-750/assignments-mpudil/tree/classification-tree-3/classification-tree/Timing_Profiling) folder includes code used to test the speed of the regression tree 
along with relevant graphs that plot the number of rows/columns in the data against the time taken 
to create the tree.


The [classification_tree.py](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/classification_tree.py) module contains the functions, classes, and methods necesary to grow, prune, and predict a regression tree given some training data. 

The [random_forest.py](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/random_forest.py) module includes the code for creating a random forest and using
a random forest for predictions.

The [sql_tree.py](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/sql_tree.py) module includes the code for creating a decision tree from
data that is stored in SQL and using it for predictions.

The [test_classification_tree.py](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/test_classification_tree.py) file includes tests for the classification tree.

The [test_random_forest.py](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/test_random_forest.py) file includes tests for the random forest.

The [test_sql_tree.py](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/test_sql_tree.py) file includes tests for SQL trees.


## User's Guide to Creating Classification Trees and Random Forest (General Information)

The following instructions explain the functionality of the classification trees and random forest as implemented in the [classification_tree.py](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/classification_tree.py) module. 

### Installations Necessary for Functionality

#### Python 3
This code was built using Python 3. Therefore, in order to ensure full functionality of the classification trees and random forest using Pandas dataframes as training or tests, it is necessary to use Python 3. Installation is available for free on Python's [website](https://www.python.org/downloads/)

#### PostgreSQL and associated credentials
While the sqlconnect function in the [sql_tree.py](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/sql_tree.py) module returns a cursor and runs functions that calculate impurity functions necessary to create the decision tree, it is necessary to have PostgreSQL installed and have associated credentials. This includes a host, database, username, and password. Note that these credentials must be passed through the sqlconnect functions to gain access to the aforementioned impurity functions. Click [here](https://www.postgresql.org/download/) to download PostgreSQL.

### Creating Python Decision Trees and Random Forest

Decision trees are structural mappings of binary decisions that lead to a decision about the class of an object. Here, we are primarily interested in using decision trees, and collections of decision trees (i.e. Random Forest) to model and predict numeric data that can be classified into two groups (0 and 1).

The format of the input data for Python Decision Trees and Random Forests are similar. In both cases, it is necessary to have a Pandas dataframe for both the X and Y data that will be used to train the decision tree or random forest. 

In general, the x dataframe (whether used for training or predicting) must be an nxk dataframe with column names that follow conventional Pandas dataframe syntax. It must contain and only contain the columns used as predictors for y. Each of the columns in the x dataframe must be either type int64 or float64 (i.e. no strings are allowed). The y_train dataframe must be an nx1 Pandas dataframe of 0's and 1's, where n is equal to the number of rows in the x_train dataframe (i.e. both the x_train and y_train dataframes must have the same number of rows). 

#### Creating and Using Python Decision Trees
The Python decision tree class is relatively flexible in design in that processes such as pruning, cross-validation, and determining misclassification costs and impurity functions can be altered by the user. The inputs to create a classification tree include: the x_train and y_train datasets as explained previously, and the criterion used to calculate the impurity function. The criterion is defaulted to be bayes error, but can also be Gini index (via criterion="gini"), cross entropy (via criterion="cross_entropy") or a user-defined criterion set as a function of p that follows Python syntax (e.g. criterion="p*(1-p)^2). 


Pruning the decision tree after creating it is technically optional, but highly recommended to reduce variance (you can read more about the bias and variance tradeoff [here](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229). Pruning the decision tree is performed via cross-validation of a list of several potential alphas after first constructing the decision tree. That is, the user can choose which alpha(s) to use to prune the tree, and how many cross-validation folds to include in the proccess. This is performed via the prune method. Explicit documentation of this method is shown in the [classification_tree.py](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/classification_tree.py) module, but in general, the prune function is performed by running tree.prune(folds, alphas), where tree is the DecisionTree object created. The default for pruning is to use cross-validation with 5 folds over the following set of alphas: [0.1, 0.5, 0.7]. 

After creating (and ideally, pruning) the DecisionTree, the tree can be used to perform predictions. This is done through the tree.predict method, which takes in a new_x argument: a dataframe that follows the same stucture as the x_train dataframe (including column names), but with new observations. That is, the new_x is an nxk matrix where k is equal to the number of columns in the x_train dataframe.

As a recap, the general syntax to creating a decision tree and then using it for predictions is:
```
import classification_tree as ct
tree = ct.DecisionTree(x_train, y_train) 
tree.prune(folds, alphas) 
tree.predict(x_new) 
```

#### Creating and Using Python Random Forests

Due to complexity in randomly sampling rows and columns from SQL databases, Random Forests are only available with Pandas dataframes, not SQL. To create a random forest, the user is required to have at minimum an x_train dataframe and a y_train dataframe, with the same restrictions as explained above. The user may also choose the number of folds to use in cross-validation for each tree along with the alphas (same default as with trees), the criterion (again, defaulted to bayes_error), the number of trees (default 5), the number of features to use to create the trees (randomly selected, default to 1 if k is 1 else k-1), and the sample size (default is ceiling of n/2). 

Note that trees of the random forest are stored as arrays and are accessible through the .trees() method after creating the random forest. The Python random forests can be used for prediction as well via the .predict(new_x) method, which follows the same rules as the DecisionTree's .predict method.

As a recap, the general syntax to creating a random forest and then using it for predictions is:

```
import random_forest as rf
forest = rf.RandomForest(x_train, y_train)
forest.predict(x_new)
```
More information regarding Random Forests is explained below in the Additional Details section.




### Creating and Using SQL Decision Trees

This package additionally allows users to create decision trees using tables from SQL. As mentioned, it is crucial that the user has access to a SQL database as one will not be provided to them to maintain privacy. Specifically, the user must have ready the following credentials: host, database, user, password. Once this has been achieved, the [sql_tree.py](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/sql_tree.py) module includes a sqlconnect function that allows the user to create a psycopg2 cursor that will be used as an input to create the SQLTree.

Additionally, the [impurity_functions.sql](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/Resources/impurity_functions.sql) file includes functions necessary for determining impurities that the user should execute before attempting to create a SQL classification tree.

The user should also have a table that includes all of the X's and the y the user would like to use. It is possible that the table has additional columns unlike the original DecisionTree or RandomForest classes for Pandas. However, the user must list the names of the x columns and y column that will be used to create the SQLTree, along with the table name and cursor. The table name cannot be a SQL command name or include punctuation. The SQL Tree can be created via:
```
SQLTree(table_name, x_names, y_name, cur)
```

Optionally, the user can add criterion, which is defauted to "bayes_error" but can also be "gini" or "cross_entropy".

Once the user has created the SQL Tree, it can be pruned using a specific alpha via the .prune(alpha) method. Note here that cross-validation is not used to choose among a list of best alphas, again due to the difficulty in random selection in SQL tables. The main difference here is that the individual nodes in the SQL tree do not hold data themselves, but only the paths that led to the node being created.


### Getting Data Within Region

Functions in the [classification_tree.py](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/classification_tree.py) and [sql_tree.py](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/sql_tree.py) allow users to access data given specific paths or cutoffs. To subset data from dataframes from a particular path, use the get_data_in_region(x_df, y_df) which subsets the dataframe to only include the x and y associated with (a) particular path(s). For example, executing 

```
get_data_in_region(x_df, y_df, path = ['var1 < 5', 'var2 > 8'])
```

will return the x dataframe and corresponding y dataframe where the path is true. Similarly, the get_data_in_region function in [sql_tree.py](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/sql_tree.py#L8) allows the user to run the SQL query to select all data that matches a particular path (format the same as for dataframes) with the option of fetching the results via fetch=True. 




## Additional Details Regarding Functionality of Decision Trees and Random Forests in Python and SQL

This section is dedicated to providing an extensive description of the functionalities of the Python and SQL decision trees and Python Random Forests. Specifically, we focus on the classes created (Node, Decision Tree, Random Forest, SQLNode, and SQLTree) and their associated methods. 

### Functionality of Nodes and Decision Trees in Python

Decision trees are essentially hierarchical collections of nodes where each node, in our case, holds several pieces of information: the data, the path that led to its creation, the criterion used (phi), and the number of rows in the original dataset (used to determine misclassification cost). Now, the Node class itself is much more of a helper class, so it is not likely that the user will need to call the Node() class. However, this class is helpful when creating the DecisionTree class. Because of this, the explanation of the Node class is most clearly explained when considering the building of a DecisionTree class.

In the init state of the DecisionTree class, which also takes the x and y dataframes as inputs (along with potentially the criterion), the Node class is called to create the parent node, with the same data and criterion as the DecisionTree. When the Decision Tree calls to create the parent node, the Node class first tries to make a left and right node (called self.lhs and self.rhs) by determining the impurity of each potential split. Each potential split is made up of two things: the variable we would like to split on (called xj in the class) and the value at which we would like to make the cutoff (called S). The method potential_splits takes in a particular xj and determines the possible S's by taking the unique values of the column. Both the best split and the cutoff help determine the impurity reduction.


In general, the impurity is a function phi that can either by bayes error = min(p, 1-p), cross entropy = -p*log(p) - (1-p) * log(1-0), or Gini = p(1-p), where p is the proportion of ones in the node. In order to determine the best split, we find the combination of xj and S that lead to the greatest difference in impurity. This looping over the variable and cutoff is performed in the best_split method. This difference is calculated by the following formula:

▲I(s, A) = I(A) - p<sub>L</sub>I(A<sub>L</sub>) - p<sub>R</sub>I(A<sub>R</sub>)

where p<sub>L</sub> is the fraction of observations that fall into the left side, i.e. where xj  $\le$ S, and similarly p<sub>R</sub> is the fraction of observations that fall into the right right side, i.e. where xj > S. I(A<sub>L</sub>) is the impurity of the left side and I(A<sub>R</sub>) is the impurity of the right side. The impurity options include gini, cross_entropy, and bayes_error, but may also be a string defined by the useras a function of p, such as "p(1-p)^2. While no particular rules are applied to the function, it is most common that the function has its maximum at p = 1/2 and its minimum at p=0 and p=1, so pure regions have all their data points in the same class and impure regions have half and half. The impurity_reduction method finds the associated impurity for the specified criterion measure.


After considering all potential splits, we take the largest split and determine if it is greater than 0. If it is, we make the split, and if not, then we do not make any splits. We interatively perform this process such that the child nodes can have children, etc. Note that the only way that the best split will not be greater than 0 is when the entire decision tree completely finishes categorizing the data into 0's and 1's perfectly. At this point, the decision tree has been completely made, but is not pruned. The nodes that are at the bottom of the decision tree are called leaves, and the property is_leaf determines whether a node is or is not a leaf. 

Once the tree has been created, the user can return the subsetted x or y dataframe on any particular node by calling something like tree.lhs.x_train, where tree is the parent node in the Decision Tree, lhs, corresponds to the paren't lhs child, and the x_train is the portion of the data the user provided in init for that particular node. Similarly the user can call .y_train for the associated y data, or .path to find the path that led to the creation of the node. 

Next, we need to prune the tree, assuming we want a tree that makes good predictions on test data[^1] 
[^1]: i.e. is not overfit. 

In order to do this, we first determine whether the children node we are looking at are leaves. If they are not, then we continue down the tree. If they are, then we look at the difference between the misclassification cost in the case where the node was turned into a leaf versus if it is not. If that difference is less than the alpha star chosen by k-fold cross-validation, then we prune the tree by setting self.lhs and self.rhs equal to None. If the children had children, then doing so will make them disappear as well. 

Once we have pruned all the nodes for which the difference in misclassification cost is less than the alpha star, we are ready to use the Decision Tree for prediction. On the user's side, all that is required is a new_x dataframe that has the same columns (and follows the same rules) as the original x_train dataframe used to construct the tree. When the user feeds in the new_x dataframe, the DecisionTree predict method queries the Node class prediction function in its predict method, which then takes one single observation row, finds the node that matches the data, and returns the (rounded) mean of the y's in that node using the predict_row method of the Node class. The process of taking one observation and determining which node it belongs to and taking the node's mean repeats for every row in the test dataset is performed for every observation in the test dataset. The results for each observation are then appended into a dataframe of predictions. 

It is also possible to ensure that the tree is valid using the .is_valid property. This property consists of querying 3 functions, all of which are aimed and ensuring that the tree follows particular rules:

1. No nodes are empty. This is calculated using the no_nodes_empty method in the DecisionTree class. The algorithm uses recursion to traverse the tree, starting with the parent node, checking each node to make sure it has data (y values). If it finds a node that is empty, it returns False, otherwise it returns True.

2. Correct splits. Next, we need to make sure that if a node split on the variable xj, then all data points in self.lhs have xj <= S and all points in rhs has xj > S. The correct_splits method uses a similar recursion method as in (1), we check to see if all data points in the node have xj <=S or all have xj > S. 

3. The summed data of the leaves matches the data in the parent node. This is performed via the correct_nleaves method, which uses a similar recursion method as in (1) and (2), but adds up the number of rows in y and once all the nodes have been accounted for, it checks to make sure that the number of rows in y for the leaves match the number of rows in y for the parent node.

### Functionality of Random Forests
A random forest is a collection of decision trees with the same properties as explained above, with the exception that rather than using the entire dataset, each of the trees use a subset of the data. Specifically, the rows of the data are sampled via bootstrapping (i.e. with replacement) whereas the columns of X are sampled via permutation (i.e. without replacement). See [here](https://towardsdatascience.com/understanding-random-forest-58381e0602d2) for further explanations regarding Random Forests. 

The RandomForest class, available [here](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/random_forest.py), is constructed by creating trees from the DecisionTree class via the build_tree method. To create a random forest, the user must call the RandomForest class. As mentioned earlier in the "Creating and Using Python Random Forests" section above, they are required to input at minimum the x_train and y_train dataframe, and optionally the number of folds, alphas, criterion, n_trees, n_features, and sample_size. The init then calls the build_tree function and creates an array where each item is a DecisionTree with the specified attributes.

The RandomForest class also has prediction capabilities, used in a similar way as the Decision Trees. The RandomForest method "predict" calls helper function/method "predict_row" for each row in the new_x dataset provided by the user. For each row, it calls each tree's predict function to classify the row. Once all of the trees return a prediction (see the above section for a recap on how this is performed), the Random Forest returns a dataframe of predictions to the user.


## Functionality of SQL Decision Trees 
The SQLNode and SQLTree classes are subclasses of the Node and DecisionTree superclasses, respectively. The overall functionality and design of either class are quite similar. The main differences here are:
1. The SQLTree class is (obviously) used for data stored in SQL. It is flexible in that it can easily process larger tables with millions of observations.
2. The nodes do not hold the actual data. Rather, they contain the path(s) that were taken to arrive at the node. These paths are used to make predictions through the following algorithm (see the "predict" methods for the SQLTree and SQLNode classes in the [sql_tree.py](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/sql_tree.py) module:
    1.  Find leaf node by traversing tree through recursion 
    2. Determine what y is in the node
    3. Update all rows in "pred" column of new table where the test column values match 
    Note that this algorithm differs from the original DecisionTree class in that rather than taking each observation and traversing the tree each time, it only traverses the tree once. Specifically, whenever it finds a leaf, it then fills in observations that match the path of the leaf with the mean y value of that leaf. This is functionally equivalent to the DecisionTree method (although faster for large datasets, especially with many splits)

3. The specific alpha level for pruning must be chosen by the user rather than through cross-validation. This has to do with the complexity of sampling in SQL. 

4. The inputs into the SQLTree class vary from the original DecisionTree class (see "Creating and Using SQL Decision Trees"). 

## Benchmark
The [Timing_Profiling](https://github.com/36-750/assignments-mpudil/tree/classification-tree-3/classification-tree/Timing_Profiling) file holds [code](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/Timing_Profiling/Benchmark.py) and graphics that have been used to obtain estimates about how the time it takes to create, prune, and predict using the classification trees, as well as how long it takes to create and predict on Random Forests with different sizes (rows and columns) of the data. In general, it was found that for small datasets, where rows*columns is less than 50, all methods take less than a minute to work. However, for moderate to large-sized data sets, pruning can take upwards of 15 minutes or so: about 6-8 times longer than creating the decision tree itself. This suggests need to use tools like cython to improve the time it takes to perform these processes since there are many moving parts and long processes that are taking place for each prune or for the creation of each node. 

As shown in the [Longest_Functions.txt](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/Timing_Profiling/Longest_Functions.txt) file, it takes longest to determine the impurity reduction of the potential best splits because it uses a double for loop to loop through every unique value of every column in the X matrix and determine the impurity. This could be improved by using cython to improve speed, which will be implemented in future versions. 

## Overview of Tests
This section lists the major tests that have been performed to test the functionality, speed, and accuracy of the Decision Trees and Random Forests.

### Decision Tree Tests
Tests for the DecisionTree class are found [here](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/test_classification_tree.py). The major tests can be grouped into the following 7 categories:
1. Ensuring unpruned DecisionTrees of various column lengths perfectly predict the training datasets used to initially create the tree and can correctly predict new data.
2. Ensuring that each tree in the cases above are valid (see
[this section](###Functionality-of-Nodes-and-Decision-Trees-in-Python) for clarification of what "valid" means). Also, trees with random data also end up being valid.
3. Ensuring an assertion error arises under the following circumstances:
    1. The y_train dataframe is not made up of 0's and 1's.
    2. There are missing values in either the x_train or y_train data
    3. There are incorrect inputs into the DecisionTree 
4. A warning ensues when any column in x_train has only one value
5. If two possible splits reduce the impurity by the same amount, then the one that was tried first is picked. Note here that "first" is by column then by row as it appears in the x_train dataframe). 
6. The classifier yields the correct answer on contrived datasets where all Y's are equal in a certain range
7. Miscellaneous tests related to the functionality of the specific methods of the DecisionTree class

### Random Forest Tests
Because of the randomness involved in the creation of the random forest, and because we have already made several tests to ensure that the decision trees are correct, the random forest tests found in the [test_random_forest.py](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/test_random_forest.py) file contain fewer tests than the decision tree tests. However, it was still possible to produce the following tests:

1. Ensuring the random forest produces the correct number of trees with the correct dimensions
2. Ensuring that each tree the random forest makes is valid
3. Ensuring that the random forest predicts the training data with high accuracy for any random dataset (that follows the x_train and y_train rules)

Along with these basic tests, the [Empirical folder](https://github.com/36-750/assignments-mpudil/tree/classification-tree-3/classification-tree/Empirical) uses real data to determine the accuracy of the Random Forests and Decision Trees under various circumstances.

### SQL Tree Tests
The [test_sql.py](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/test_sql_tree.py) file includes several tests that were performed to ensure that the SQL Decision Trees were produced correctly. Because the SQLTree class is a subclass of DecisionTree, it was not necessary to test all methods used in the SQLTree. However, the following tests were possible to perform without being redundant:

1. Ensuring that predictions on the same dataset are the same for SQLTrees and DecisionTrees and that the paths taken are the same.
2. Ensuring that the get_data_in_region function returns the correct subset of the data in SQL and matches the results from the classification tree get_data_in_region function.
3. Ensuring the SQL tree perfectly predicts the training dataset (without pruning)

The first test was performed by randomly generating data (i.e. an x_train, y_train, and x_test dataset), feeding it into both dataframes and SQL tables, creating the trees using the training dataset, and then using the x_test dataset to determine predictions. It was found that both the SQL tree and the Python tree produce the same results. The second test was performed by creating sample data, creating a path that would ensure some points were included and others left out, and then making sure that both methods arrived at the same subset of points, which they did. The third test was perfomed by creating a SQL table with multiple predictor columns, making sure to not prune the tree, and then using the same data to make sure it resulted in predictions that were the same as the original outcome variable.


### Empirical Tests
[Data](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/Empirical/acceptance.csv) regarding acceptance into Berkeley was used in order to test the accuracy and prediction performance of the decision trees and random forest under various levels of alpha. As expected, the prediction performance varied by the level alpha that was used in the case of the Decision Tree, and by the number of trees and subsetting method (rows and columns per tree) when using random forests. The [analysis.txt](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/Empirical/analysis.txt) file explains the performance of the decision trees and random forest with the code used to run the empirical tests [here](https://github.com/36-750/assignments-mpudil/blob/classification-tree-3/classification-tree/Empirical/empirical.py). 

## Authors

* **Mitchell Pudil**