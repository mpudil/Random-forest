import numpy as np
import pandas as pd
import psycopg2 as pg
import classification_tree as ct


# Connect to SQL
def sqlconnect(host, database, user, password):
    """ Return cursor for connection to SQL database and defines SQL functions for impurity reduction
    
    Parameters
    ----------
    host: host name for connection to SQL
    database: database used for connection to SQL
    user: user for connection to SQL
    password: password for connection to SQL
    
    """
    # Assert valid connection
    conn = pg.connect(host = host, database=database, user = user, password = password)
    cur = conn.cursor()
    return cur

def get_data_in_region(table_name, cur, path, fetch=False):
    """ Execute SELECT query to subset the SQL data by path.
    
    Parameters
    ----
    path: list of paths as strings e.g. ['var1 > 3', 'var2 <= 4']
    fetch: Boolean, True if user wants to store results of output as a matrix.

    """
    if len(path)==1:
        cur.execute("SELECT * FROM " + table_name + " WHERE " + path[0] + ";")
    else:
        cur.execute("SELECT * FROM " + table_name + " WHERE " + " AND ".join(path) + ";")

    if fetch is True:
        return cur.fetchall()
    else:
        return



class SQLNode(ct.Node):
    """ A SQL Node used for classification when data is in SQL. Subclass of Node.
    
    Instances have several attributes:
    
    table_name: name of table in SQL. Must be one word. Cannot be a SQL keyword e.g. "DROP". 
        Cannot include punctuation.
        
    x_names: List names of relevant x columns to be used for classification. All columns must be type numeric.
    
    y_name: name of y column used. Note that the associated column y must be type int and only hold values 0 or 1.
    
    cur: Cursor, i.e. output from sqlconnect function.
    
    criterion: The method of impurity reduction used to determine impurity. Defaulted to "gini",
        but may also be "cross_entropy" or "bayes"
        
    path: List of paths taken to arrive at particular node, e.g. [["var1 <= 4"], ["var2 > 9]]". 
        Initialized as empty list where each element is a comparison between the cutoff variable, 
        and cutoff. 
        
    nrow: Number of rows in SQL table. Note it does not have to be provided by user, although it can.
      

    
    """
    def __init__(self, table_name, x_names, y_name, cur, criterion = "bayes_error", path = [], nrow = None, lhs = None, rhs = None): 
        """ Initialize SQL Node class. """
        
        # Create x_train and y_train
        self.table_name = table_name
        self.x_names = x_names
        self.y_name = y_name
        self.cur = cur
        self.path = path
        self.criterion = criterion
        
        if nrow is None:
            self.cur.execute("SELECT COUNT(*) FROM " + table_name + ";")
            nrow = self.cur.fetchone()[0]
        
        self.nrow = nrow
        
        self.make_children()

    def potential_splits(self, potential_xj):
        """ Return potential splits to be used to find the optimal splits for pruning 
        
        
        Parameters: 
        ----------
        
        potential_xj: Name of column in X, i.e. element of x_names list
        
        """
    
        self.cur.execute("SELECT DISTINCT " + potential_xj +  " FROM " + self.table_name + ";")
        potential_splits = [ii[0] for ii in self.cur.fetchall()]
        return potential_splits


    def impurity_reduction(self, xj, S):
        """ Determine the change in impurity for a given combination of xj and S
        
        Parameters:
        
        xj: Name of a column in X, i.e. element of x_names list
        S: cutoff value (numeric, float, or int)
        
        """
        # Determine number of rows in left and right children and calculate respective impurities for parent, 
        # left, and right 
        if len(self.path) == 0:

            self.cur.execute("SELECT COUNT(*) FROM " + self.table_name + " WHERE " + xj + " <= " + str(S) + ";")
            n_left = self.cur.fetchone()[0]

            self.cur.execute("SELECT COUNT(*) FROM " + self.table_name + " WHERE " + xj + " > " + str(S) + ";")
            n_right = self.cur.fetchone()[0]


            self.cur.execute("SELECT " + self.criterion + "(AVG(" + self.y_name + ")) FROM " + self.table_name + ";")
            I_A = float(self.cur.fetchone()[0])

            if n_left == 0 or n_right == 0:
                return 0
            else: 
                self.cur.execute("SELECT " + self.criterion + "(AVG(" + self.y_name + ")) FROM " + self.table_name + " WHERE " + xj + " <= " + str(S) + ";")
                I_L = float(self.cur.fetchone()[0])

                self.cur.execute("SELECT " + self.criterion + "(AVG(" + self.y_name + ")) FROM " + self.table_name + " WHERE " + xj + " > " + str(S) + ";")
                I_R = float(self.cur.fetchone()[0])


        else:

            self.cur.execute("SELECT COUNT(*) FROM " + self.table_name + " WHERE " + xj + " <= " + str(S) + " AND " + " AND ".join(self.path) + ";")
            n_left = self.cur.fetchone()[0]

            self.cur.execute("SELECT COUNT(*) FROM " + self.table_name + " WHERE " + xj + " > " + str(S) + " AND " + " AND ".join(self.path) + ";")
            n_right = self.cur.fetchone()[0]
        
            if n_left == 0 or n_right == 0:
                return 0
    
            self.cur.execute("SELECT " + self.criterion + "(AVG(" + self.y_name + ")) FROM " + self.table_name + " WHERE " + " AND ".join(self.path) + ";")
            I_A = float(self.cur.fetchone()[0])

            self.cur.execute("SELECT " + self.criterion + "(AVG(" + self.y_name + ")) FROM " + self.table_name + " WHERE " + " AND ".join(self.path) + " AND " + xj + " <= " + str(S) + ";")
            I_L = float(self.cur.fetchone()[0])

            self.cur.execute("SELECT " + self.criterion + "(AVG(" + self.y_name + ")) FROM " + self.table_name + " WHERE " + " AND ".join(self.path) + " AND " + xj + " > " + str(S) + ";")
            I_R = float(self.cur.fetchone()[0])

                
        # Calculate change in impurity
        frac_left = n_left / (n_left + n_right)
        frac_right = n_right / (n_left + n_right)

        change_impurity = I_A - frac_left*I_L - frac_right*I_R
        
        return change_impurity
        

    def best_split(self):
        """ Return best split by iterating through all potential splits and X columns 
        Chooses largest change in impurity (ir) if it exists.
        """
        best_splits = [[0, None, None]]
        impurity, best_S, best_xj = 0, None, None
        
        for xj in self.x_names:
            for S in self.potential_splits(xj):
                ir = float(self.impurity_reduction(xj, S))
                if ir > impurity:
                    impurity, best_S, best_xj = ir, S, xj
                    best_splits.append([S, xj])
                else: 
                    pass
        
        return best_S, best_xj

    
    def make_children(self, criterion="bayes_error"):
        """ Create children 
        
        Parameters:
        
        xj: Name of a column in X, i.e. element of x_names list
        S: cutoff value (numeric, float, or int)
        
        """
        S, xj = self.best_split()
      
        if S is None and xj is None:
            self.lhs = None
            self.rhs = None
            return
        

        # Note that here, we are not storing the actual data, but just the paths, unlike the regular Node class.
    
        self.lhs = SQLNode(self.table_name, self.x_names, self.y_name, self.cur, self.criterion, self.path + [xj + ' <= ' + str(S)], self.nrow) 

        self.rhs = SQLNode(self.table_name, self.x_names, self.y_name, self.cur, self.criterion, self.path + [xj + ' > ' + str(S)], self.nrow)
        
        return

    
    
    def predict(self, new_table):
        """ Classify observations from a new table with same column names and types.
        Requires table to have an empty y column of type int that will be filled out through 
        recursion. New table must be accessible with the same credentials/cursor as the original table.
        
        Algorithm: 
        
        1. Find leaf node by traversing tree through recursion 
        2. Determine what y is in the node
        3. Update all rows in y column of new table where the test column values match 
        
        Parameters:
        
        new_table: Name of new table. Name must follow same rules as table_name
         
        """
        
        if self.is_leaf:
            if len(self.path) == 1:
                self.cur.execute("SELECT ROUND(AVG(" + self.y_name + ")) FROM " + self.table_name + " WHERE " + self.path[0] + ";")
            else:
                self.cur.execute("SELECT ROUND(AVG(" + self.y_name + ")) FROM " + self.table_name + " WHERE " + " AND ".join(self.path) + ";")
            y = self.cur.fetchone()[0]
            self.cur.execute("UPDATE " + new_table + " SET preds = " + str(y) + " WHERE " + " AND ".join(self.path) + ";")
            
        else:
            self.lhs.predict(new_table)
            self.rhs.predict(new_table)
        


    def diff_misclass_cost(self):
        """Calculate parent and children misclassification cost and then determine their difference 

        """
        
        if len(self.path) == 0:
            # Determine correct y
            self.cur.execute("SELECT ROUND(AVG(" + self.y_name + ")) FROM " + self.table_name + ";")
            correct_y = int(self.cur.fetchone()[0])

            # Determine length of y
            self.cur.execute("SELECT COUNT(*) FROM " + self.table_name + ";")
            y_length = int(self.cur.fetchone()[0])

            # Determine how many y's are wrong
            self.cur.execute("SELECT COUNT(*) FROM " + self.table_name + " WHERE " + self.y_name + " <> " + str(correct_y) + ";")
            wrong_parent = self.cur.fetchone()[0]
            
        else:
            # Determine correct y for node
            self.cur.execute("SELECT ROUND(AVG(" + self.y_name + ")) FROM " + self.table_name + " WHERE " + " AND ".join(self.path) + ";")
            correct_y = int(self.cur.fetchone()[0])

            # Determine length of y node
            self.cur.execute("SELECT COUNT(*) FROM " + self.table_name + " WHERE " + " AND ".join(self.path) + ";")
            y_length = self.cur.fetchone()[0]

            # Determine how many y's are wrong in node
            self.cur.execute("SELECT COUNT(*) FROM " + self.table_name + " WHERE " + self.y_name + " <> " + str(correct_y) + " AND " + " AND ".join(self.path) + ";")
            wrong_parent = self.cur.fetchone()[0]

        # Calculate misclassification for parent
        misclass_parent = wrong_parent / y_length

        # Calculate fraction of node's length to length of entire dataset
        frac_all_parent = y_length / self.nrow
        
        # Determine R(t): misclassification cost
        misclass_cost_parent = misclass_parent * frac_all_parent


        # Calculating Children Misclass Cost
        # Left

        self.cur.execute("SELECT ROUND(AVG(" + self.y_name + ")) FROM " + self.table_name + " WHERE " + " AND ".join(self.lhs.path) + ";")
        correct_y_left = self.cur.fetchone()[0]
        
        self.cur.execute("SELECT COUNT(*) FROM " + self.table_name + " WHERE " + " AND ".join(self.lhs.path) + ";")
        left_length = self.cur.fetchone()[0] 
     
        self.cur.execute("SELECT COUNT(*) FROM " + self.table_name + " WHERE " + self.y_name + " <> " + str(correct_y_left) + " AND " + " AND ".join(self.lhs.path) + ";")
        misclass_left = self.cur.fetchone()[0] / left_length

        frac_all_left = left_length / self.nrow
        
        misclass_cost_left = (misclass_left/left_length) * frac_all_left


        # Right

        self.cur.execute("SELECT ROUND(AVG(" + self.y_name + ")) FROM " + self.table_name + " WHERE " + " AND ".join(self.rhs.path) + ";")
        correct_y_right = self.cur.fetchone()[0]
        
        self.cur.execute("SELECT COUNT(*) FROM " + self.table_name + " WHERE " + " AND ".join(self.rhs.path) + ";")
        right_length = self.cur.fetchone()[0] 


        self.cur.execute("SELECT COUNT(*) FROM " + self.table_name + " WHERE " + self.y_name + " <> " + str(correct_y_right) + " AND " + " AND ".join(self.rhs.path) + ";")
        misclass_right = self.cur.fetchone()[0] / right_length

        frac_all_right = right_length / self.nrow
        
        misclass_cost_right = misclass_right/right_length * frac_all_right   

        # Total misclass cost for children
        misclass_cost_children = sum([misclass_cost_left, misclass_cost_right])
        mcost = misclass_cost_parent - misclass_cost_children  

        return mcost 



    # Note here that the prune function will/should work the same for the node class, so it is not necessary to overwrite that function.

class SQLTree(ct.DecisionTree):
    
    """ A SQL Tree used for classification when data is in SQL. Subclass of DecisionTree.
    
    Instances have several attributes:
    
    table_name: name of table in SQL. Must be one word. Cannot be a SQL keyword e.g. "DROP". 
        Cannot include punctuation.
        
    x_names: List names of relevant x columns to be used for classification. All columns must be type numeric.
    
    y_name: name of y column used. Note that the associated column y must be type int.
    
    cur: Cursor, i.e. output from sqlconnect function.
    
    alpha: Alpha to be used for pruning. Note that no cross-validation will be performed. Defaults to 0.5.
    
    criterion: The method of impurity reduction used to determine impurity. Defaulted to "bayes_error",
        but may also be "cross_entropy" or "gini"
    """

    
    def __init__(self, table_name, x_names, y_name, cur, criterion = "bayes_error", path = []):
        """ Initialize SQLTree with user-provided inputs """
        # Create x_train and y_train
        
        self.table_name = table_name
        self.x_names = x_names
        self.y_name = y_name
        self.cur = cur
        self.path = path
        self.criterion = criterion
        self.tnode = SQLNode(self.table_name, self.x_names, self.y_name, self.cur, self.criterion)
        
        if len(self.path) == 0:
            assert ";" or "drop" or "select" or ";" or "," or "." not in self.table_name, "Invalid table name"

    def predict(self, new_table_name):
        """Predict entire Y matrix for all X's

        Parameters
        ----------
        
        new_table: Name of new table, but with only X columns (and same X column names/types as original table).
        
        Note that none of the column names in new_table can be called "preds" since that names is reserved for 
        the predictions and will be added onto the table.
        
        """
        assert new_table_name != self.table_name, "New table cannot be the same as table_name from __init__"
        self.cur.execute("ALTER TABLE " + new_table_name + " ADD COLUMN preds INT;")
        return self.tnode.predict(new_table_name)


    def prune(self, alpha=0):
        """ Prune tree
        
        Parameters
        ----------
        
        alpha: alpha level to prune trees
        
        
        """
        
        self.tnode.prune(alpha)
    
