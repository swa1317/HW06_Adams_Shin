import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math
import time

################################
#                              #
#   Bahdah Shin                #
#   Samuel Adams               #
#   10/14/2020                 #
#                              #
################################

# PARAMETERS
START = time.time()

class Node:
    """
    The Node is the building block of the decision tree.
    predicted_class: provides the class prediction.
                    Will only handle positive integers. -> 'Assam': 1, 'Bhuttan': 0
    attribute_index: otherwise known as column index from the 2d data
    threshold: the number that divides the data.
                If the dataset is bool(0, 1), then it's set to 0.5.
                Numeric dataset has calculated threshold.
    left: left node
    right: right node
    if both left and right nodes are None, the main node is a leaf node.

    """
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.attribute_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

def csv_to_array(csv_filename):
    """
    csv_filename: the name of the csv file
    return: the numpy 2d array of csv file
    """
    # Bhuttan is set to 0 instead of -1 due to restrictions on certain function
    result_mapping = {'Assam': 1, 'Bhuttan': 0}
    # reads the csv file to pandas DataFrame Object
    csv_pandas = pd.read_csv(csv_filename)
    # capturing the data with specific column
    attributes = csv_pandas[['Age','Ht','TailLn','HairLn','BangLn', 'Reach', 'EarLobes']].copy()
    # quantizing the data
    attributes['Age'] = attributes['Age'].apply(lambda x: quantize(x, 2))
    attributes['Ht'] = attributes['Ht'].apply(lambda x: quantize(x, 5))
    attributes['TailLn'] = attributes['TailLn'].apply(lambda x: quantize(x, 1))
    attributes['HairLn'] = attributes['HairLn'].apply(lambda x: quantize(x, 1))
    attributes['BangLn'] = attributes['BangLn'].apply(lambda x: quantize(x, 1))
    attributes['Reach'] = attributes['Reach'].apply(lambda x: quantize(x, 1))

    # converting it to numpy
    attributes = attributes.to_numpy()
    result = csv_pandas[['Class']].copy()
    result = result['Class'].map(result_mapping)
    result = result.to_numpy()
    # combine the attribute data and true class data into one. The true class is added at the end
    data = np.concatenate((attributes, result.reshape(-1, 1)), 1)
    return data.astype(int) # convert the data value type to int

def quantize(x, base=1):
    """
    x: some value
    base: the value to quantize or bin
    return: a quantized value
    """
    return float(base * math.floor(x/base))

def _sort_min_by_col(data, col_index):
    """
    data: the 2d numpy dataset that includes the true class column at the end
    col_index: the attribute index number for the 2d numpy dataset
    return: Sort the specified column of the 2D array called data and return the sorted array
    """
    # argsort returns the indices that would sort an array, so applying them to the array called data will create the
    # sorted array
    return data[np.argsort(data[:, col_index])]  # data[:, col_index] gets the data from the specified column index

def get_col_data(data, col_index):
    """
    data: the 2d numpy dataset that includes the true class column at the end
    col_index: the attribute index number for the 2d numpy dataset
    return: the array of data at the specified column index for the 2D array called data
    """
    return data[:,col_index] # gets the data from col_index only

def get_gini_from_threshold(data, col_index, threshold):
    """
    data: the 2d numpy dataset that includes the true class column at the end
    col_index: the attribute index number for the 2d numpy dataset
    threshold: the number that divides the dataset
    return: the lowest gini value
    """
    below_thresh_positive = 0
    below_thresh_negative = 0
    above_thresh_positive = 0
    above_thresh_negative = 0
    gini = 1
    for row in range(data.shape[0]):
        if data[row][col_index] >= threshold:
            if data[row][-1] > 0:  # +1 -> Assam
                above_thresh_positive += 1
            else:  # -1 -> Bhuttan
                above_thresh_negative += 1
        else:
            if data[row][-1] > 0:  # +1 -> Assam
                below_thresh_positive += 1
            else:  # -1 -> Bhuttan
                below_thresh_negative += 1

        below_thresh_total = below_thresh_positive + below_thresh_negative
        above_thresh_total = above_thresh_positive + above_thresh_negative
        if below_thresh_total > 0 and above_thresh_total > 0:
            thresh_total = below_thresh_total + above_thresh_total

            gini_above_thresh = 1 - \
                        (above_thresh_positive / above_thresh_total) ** 2 - \
                        (above_thresh_negative / above_thresh_total) ** 2
            gini_below_thresh = 1 - \
                         (below_thresh_positive / below_thresh_total) ** 2 - \
                         (below_thresh_negative / below_thresh_total) ** 2
            gini = (above_thresh_total / thresh_total) * gini_above_thresh + \
                   (below_thresh_total / thresh_total) * gini_below_thresh
        else:
            gini = 1
    return gini

def get_lowest_gini_on_col_numeric(data, col_index):
    """
    data: the 2d numpy dataset that includes the true class column at the end
    col_index: the attribute index number for the 2d numpy dataset
    return: best_gini -> the lowest gini value
            best_threshold -> calculated from gradient decent
            col_index -> the attribute index
    """
    # find the best threshold based on gini value
    # get lowest gini value
    data = _sort_min_by_col(data, col_index)
    min_from_col = data[0][col_index]
    max_from_col = data[-1][col_index]

    best_gini = 1
    best_threshold = 0
    n_steps = 5
    full_range = max_from_col - min_from_col
    increment = full_range / (2 * n_steps + 1)
    increment_stop = 0.25
    while increment > increment_stop:
        old_best_threshold = best_threshold
        min_threshold_val = old_best_threshold - (n_steps * increment)
        max_threshold_val = old_best_threshold + (n_steps * increment)
        for threshold_value in np.arange(min_threshold_val, max_threshold_val, increment):
            old_threshold = best_threshold
            old_gini = get_gini_from_threshold(data, col_index, old_threshold)
            new_threshold = threshold_value
            new_gini = get_gini_from_threshold(data, col_index, new_threshold)

            if new_gini > old_gini:
                best_threshold = old_threshold
                best_gini = old_gini
            else:
                best_threshold = new_threshold
                best_gini = new_gini
        learning_rate = 40/64
        increment = increment * learning_rate
    return best_gini, best_threshold, col_index

def get_lowest_gini_on_col_bool(data, col_index):
    """
    data: the 2d numpy dataset that includes the true class column at the end
    col_index: the attribute index number for the 2d numpy dataset
    return: gini -> the lowest gini value
            threshold -> 0.5 constant
            col_index -> the attribute index
    """
    zero_positive = 0
    zero_negative = 0
    one_positive = 0
    one_negative = 0
    for row_index in range(data.shape[0]):
        if data[row_index][col_index] == 0:
            if data[row_index][-1] > 0:
                zero_positive += 1
            else:
                zero_negative += 1
        else:
            if data[row_index][-1] > 0:
                one_positive += 1
            else:
                one_negative += 1

    zero_total = zero_positive + zero_negative
    one_total = one_positive + one_negative
    if zero_total > 0 and one_total > 0:
        both_sides_total = zero_total + one_total
        # gini impurity on each nodes
        gini_zero = 1 - ((zero_positive)/(zero_total))**2 - ((zero_negative)/(zero_total))**2
        gini_one = 1 - ((one_positive)/(one_total))**2 - ((one_negative)/(one_total))**2

        # weighted average of gini impurities
        gini = (zero_total/both_sides_total)*gini_zero + (one_total/both_sides_total)*gini_one
    else:
        gini = 1
    # split is zero and one
    # return threshold as 0.5
    threshold = 0.5
    return gini, threshold, col_index

def train(data, max_depth, min_data_records, node_purity):
    """
    data: the 2d numpy dataset that includes the true class column at the end
    return: the decision tree
    """
    return _grow_tree(data, max_depth, min_data_records, node_purity)

def _grow_tree(data, max_depth, min_data_records, node_purity, depth=0):
    """
    data: the 2d numpy dataset that includes the true class column at the end
    depth: the depth of the decision tree
    return: the decision tree

    Recursively builds a decision tree.

    max depth is 6
    min data size is 10
    accuracy threshold is 0.95
    """


    if depth < max_depth and data.shape[0] >= min_data_records:
        # np.unique -> gets unique elements in array
        # values: sorted unique values
        # count: sorted unique value frequency
        (values, counts) = np.unique(get_col_data(data, -1), return_counts=True)
        # np.argmax -> Returns the indices of the maximum values.
        values_and_counts_index = np.argmax(counts)
        # sum() -> summation of list
        predicted_class_accuracy = counts[values_and_counts_index] / sum(counts)

        if predicted_class_accuracy < node_purity:
            predicted_class = values[values_and_counts_index]
            node = Node(predicted_class=predicted_class)

            number_of_attributes = data.shape[1] - 1

            gini_value_and_threshold_list = []
            for col_index in range(number_of_attributes):
                unique_numbers = np.unique(get_col_data(data, col_index))
                if len(unique_numbers) == 2:
                    # bool
                    gini_value_and_threshold_list.append(get_lowest_gini_on_col_bool(data, col_index))
                else:
                    # numeric
                    gini_value_and_threshold_list.append(get_lowest_gini_on_col_numeric(data, col_index))
            # sort() -> sorts the list
            gini_value_and_threshold_list.sort(key=lambda tup: tup[0])

            # best split threshold
            node.threshold = gini_value_and_threshold_list[0][1]
            node.attribute_index = gini_value_and_threshold_list[0][2]

            data_below_threshold = np.array([])
            data_above_threshold = np.array([])

            # split the data into two

            for row_index in range(data.shape[0]):
                if data[row_index][node.attribute_index] < node.threshold:
                    data_below_threshold = np.append(data_below_threshold, data[row_index])
                else:
                    data_above_threshold = np.append(data_above_threshold, data[row_index])
            data_below_threshold = data_below_threshold.reshape(-1, 8)
            data_above_threshold = data_above_threshold.reshape(-1, 8)

            node.left = _grow_tree(data_below_threshold, max_depth, min_data_records, node_purity, depth + 1)
            node.right = _grow_tree(data_above_threshold, max_depth, min_data_records, node_purity, depth + 1)

            return node
        else:
            return None
    else:
        return None

def predict(data, tree):
    """
    data: the 2d numpy dataset. does not include the true class column at the end.
    tree: the binary tree for decision tree
    :return a list of predicted class where dataset is int
    This is the main function that calls the helper function _predict().
    """
    return np.array([_predict(single_row_of_attributes, tree) for single_row_of_attributes in data]).astype(int)

def _predict(single_row_of_attributes, node):
    """
    single_row_of_attributes: A row of attributes that does not include the true class.
    node: A object that contains threshold, attribute index, and prediction class.
          Also has left and right node.
    return: predicted class
    A recursive function that recurse into the tree.
    """
    if node.left == None and node.right == None:
        return node.predicted_class
    else:
        if single_row_of_attributes[node.attribute_index] < node.threshold:
            if node.left != None:
                return _predict(single_row_of_attributes, node.left)
            else:
                return node.predicted_class
        else:
            if node.right != None:
                return _predict(single_row_of_attributes, node.right)
            else:
                return node.predicted_class


# Given the current decision tree node, file object of classifier program, and tab index that tracks the current tab
# length for formatting, recursively traverse through each node in the decision tree and form the nested if-else system
# in the classifier program.
def recurseTreeToConditional(tree_node, file_object, tab_index):

    # tab character string of tab_index number of tabs
    tab_characters = "\t" * tab_index

    # If a leaf node
    if tree_node.left == None and tree_node.right == None:

        # If an Assam then write class_val as +1
        if tree_node.predicted_class > 0:
            command = tab_characters + "class_val=\"+1\""
        # If a Bhutan then write class_value as -1
        else:
            command = tab_characters + "class_val=\"-1\""

        # Write line of code to classifier program
        file_object.write(command + '\n')

    # If a root node
    else:

        # Since root node, write an if statement to check if the current attribute value of the node's attribute index
        # is less than the nodes threshold (signifying a need to move to the left node)
        conditional = tab_characters + "if( row[" + str(tree_node.attribute_index) + "] < " + str(tree_node.threshold) + "):"
        file_object.write(conditional + '\n')

        # if there is a left node, then we will need to make another nested if-statement so recursively call this method
        # with the left child node and tab index increased
        if tree_node.left != None:
            recurseTreeToConditional(tree_node.left, file_object, tab_index+1)

        # if there is no left child node then we have reached our prediction and will set the class_val based on if
        # the current node predicts an Assam (+1) or Bhutan (-1)
        else:
            tab_characters_nested = "\t" * (tab_index+1)
            if tree_node.predicted_class > 0:
                command = tab_characters_nested + "class_val=\"+1\""
            else:
                command = tab_characters_nested + "class_val=\"-1\""

            file_object.write(command + '\n')

        # Write the else that matches the above if statement (signifying a need to move to the right node)
        conditional = tab_characters + "else:"
        file_object.write(conditional + '\n')

        # if there is a right node, then we will need to make another nested if-statement so recursively call this method
        # with the right child node and tab index increased
        if tree_node.right != None:
            recurseTreeToConditional(tree_node.right, file_object, tab_index + 1)

        # if there is no right child node then we have reached our prediction and will set the class_val based on if
        # the current node predicts an Assam (+1) or Bhutan (-1)
        else:
            tab_characters_nested = "\t" * (tab_index + 1)
            if tree_node.predicted_class > 0:
                command = tab_characters_nested + "class_val=\"+1\""
            else:
                command = tab_characters_nested + "class_val=\"-1\""

            file_object.write(command + '\n')

# Open a new python file according to naming conventions and add the string functions below, then call the recursive
# function that writes the nested if-else statements, finish by printing and writing classification and then close
# the file object
def writeClassifierProgram(tree):
    trained_program_filename = "HW05_Classifier_Adams_Shin.py"
    file_object = open(trained_program_filename, "wt")  # create file

    file_object.write(import_string)
    file_object.write('\n')

    file_object.write(function_string)
    file_object.write('\n')

    file_object.write(predict_string)

    recurseTreeToConditional(tree, file_object, 2)

    file_object.write("\t\tprint(class_val)\n")
    file_object.write("\t\tfile_object.write(class_val + \'\\n\')\n")
    file_object.write("\tfile_object.close()\n")

    file_object.write(main_string)
    file_object.write('\n')

    file_object.close()

import_string = '''
import sys
import pandas as pd
'''

# String to write to classifier program that convert CSV to array using pandas read_csv then numpy's to_numpy method
# which produces the numpy array that the classifier program will use as its data
function_string = '''
def csv_to_array_trained(csv_filename):
\tcsv_pandas = pd.read_csv(csv_filename)
\tattributes_with_class = csv_pandas[['Age', 'Ht', 'TailLn', 'HairLn', 'BangLn', 'Reach', 'EarLobes']].copy()

\t# quantizing the data
\tattributes['Age'] = attributes['Age'].apply(lambda x: quantize(x, 2))
\tattributes['Ht'] = attributes['Ht'].apply(lambda x: quantize(x, 5))
\tattributes['TailLn'] = attributes['TailLn'].apply(lambda x: quantize(x, 1))
\tattributes['HairLn'] = attributes['HairLn'].apply(lambda x: quantize(x, 1))
\tattributes['BangLn'] = attributes['BangLn'].apply(lambda x: quantize(x, 1))
\tattributes['Reach'] = attributes['Reach'].apply(lambda x: quantize(x, 1))

\tdata = attributes_with_class.to_numpy()
\treturn data
'''

# String to write to classifier program that begins the predict method, which is a series of if-else statements
# that predict the class. The if-then statements are written using the decision tree in lines 294-330 of this file
predict_string = '''
def predict(data):
\tclassifications_filename = "HW05_Adams_MyClassifications.csv"
\tfile_object = open(classifications_filename, "wt")  # create file

\tfor row in data:
'''

# String to write to classifier that includes the main function to run, which processes the input file parameter and
# calls the csv_to_array function and the predict function
main_string = '''  
if __name__ == '__main__':
\tparameter = sys.argv[1:]
\tif len(parameter) == 0:
\t\tprint("the parameter is empty")
\telse:
\t\tparameter = parameter[0]
\t\tdata = csv_to_array_trained(parameter)
\t\tpredict(data)
'''

def check_mistakes(train_batch, test_batch, max_depth, min_data_records, node_purity):
    """
    data: the 2d data.
    return: the number of mistakes
    The data should contain the true class column at the end.
    """
    tree = train(train_batch, max_depth, min_data_records, node_purity)
    results = predict(test_batch, tree)
    true_results = get_col_data(test_batch, -1)
    (values, counts) = np.unique(true_results == results, return_counts=True)
    number_of_mistakes = counts[0] if not values[0] else counts[1]
    return number_of_mistakes

def check_depth(filename):
    MIN_DEPTH = 2
    MAX_DEPTH = (5) + 1
    MIN_DATA_RECORDS = 10
    NODE_PURITY = 0.90
    data = csv_to_array(filename)

    batch_count = 10
    batches = np.split(data, batch_count, axis=0)  # batch_count = number of batches

    error_rate = []
    print("=================================================")
    for tree_depth in range(MIN_DEPTH, MAX_DEPTH):
        number_of_mistakes = 0
        for test_index in range(len(batches)):
            test_batch = batches[test_index]
            train_indexes = []
            for train_index in range(len(batches)):
                if test_index != train_index:
                    train_indexes.append(train_index)
            train_batch = np.concatenate(list(map(lambda x: batches[x], train_indexes)), axis=0)
            number_of_mistakes = number_of_mistakes + check_mistakes(train_batch, test_batch, tree_depth, MIN_DATA_RECORDS, NODE_PURITY)
            print("Tree Depth: {0} | Test Index: {1} | TIME: {2:.2f} seconds".format(tree_depth, test_index, time.time() - START))
        error_rate.append(number_of_mistakes)

    depth_range = list(range(MIN_DEPTH, MAX_DEPTH))

    plt.bar(depth_range, error_rate)
    plt.xlabel('Depths')
    plt.ylabel('Number of Mistakes')
    plt.title('Depth Versus Mistakes')
    plt.savefig('DepthVersusMistakes.png')
    plt.show()

    print("The x value: {0}".format(depth_range))
    print("The y value: {0}".format(error_rate))

    print("=================================================")
    index_of_min_error_rate = error_rate.index(min(error_rate))
    return depth_range, error_rate, depth_range[index_of_min_error_rate]

def check_data_records(filename, best_depth):
    NODE_PURITY = 0.90
    min_data_records_list = [30, 25, 20, 15, 10, 8, 6, 5, 4, 3, 2]
    data = csv_to_array(filename)

    batch_count = 10
    batches = np.split(data, batch_count, axis=0)  # batch_count = number of batches

    error_rate = []
    print("=================================================")
    for min_data_records in min_data_records_list:
        number_of_mistakes = 0
        for test_index in range(len(batches)):
            test_batch = batches[test_index]
            train_indexes = []
            for train_index in range(len(batches)):
                if test_index != train_index:
                    train_indexes.append(train_index)
            train_batch = np.concatenate(list(map(lambda x: batches[x], train_indexes)), axis=0)
            number_of_mistakes = number_of_mistakes + check_mistakes(train_batch, test_batch, best_depth, min_data_records, NODE_PURITY)
            print("Min Data Records: {0} | Test Index: {1} | TIME: {2:.2f} seconds".format(min_data_records, test_index, time.time() - START))
        error_rate.append(number_of_mistakes)

    print("The x value: {0}".format(min_data_records_list))
    print("The y value: {0}".format(error_rate))
    # add graph here

    plt.bar(min_data_records_list, error_rate)
    plt.xlabel('Minimum Node Size')
    plt.ylabel('Number of Mistakes')
    plt.title('Minimum Node Size Versus Mistakes')
    plt.savefig('MinNodeSize.png')
    plt.show()

    print("The x value: {0}".format(depth_range))
    print("The y value: {0}".format(error_rate))

    print("=================================================")
    index_of_min_error_rate = error_rate.index(min(error_rate))
    return min_data_records_list, error_rate, min_data_records_list[index_of_min_error_rate]

def check_node_purity(filename, best_depth, best_min_data_records):
    node_purity_list = [70, 75, 80, 85, 90, 95, 96, 98]
    data = csv_to_array(filename)

    batch_count = 10
    batches = np.split(data, batch_count, axis=0)  # batch_count = number of batches

    error_rate = []
    print("=================================================")
    for node_purity in node_purity_list:
        number_of_mistakes = 0
        for test_index in range(len(batches)):
            test_batch = batches[test_index]
            train_indexes = []
            for train_index in range(len(batches)):
                if test_index != train_index:
                    train_indexes.append(train_index)
            train_batch = np.concatenate(list(map(lambda x: batches[x], train_indexes)), axis=0)
            number_of_mistakes = number_of_mistakes + check_mistakes(train_batch, test_batch, best_depth, best_min_data_records, node_purity)
            print("Node Purity: {0} | Test Index: {1} | TIME: {2:.2f} seconds".format(node_purity, test_index, time.time() - START))
        error_rate.append(number_of_mistakes)

    print("The x value: {0}".format(node_purity_list))
    print("The y value: {0}".format(error_rate))
    # add graph here

    plt.bar(node_purity_list, error_rate)
    plt.xlabel('Node Purity')
    plt.ylabel('Number of Mistakes')
    plt.title('Node Purity Versus Mistakes')
    plt.savefig('NodePurity.png')
    plt.show()

    print("=================================================")
    index_of_min_error_rate = error_rate.index(min(error_rate))

    return node_purity_list, error_rate, node_purity_list[index_of_min_error_rate]

if __name__ == '__main__':
    # filename = "Abominable_Data_HW05_v725.csv"
    parameter = sys.argv[1:]
    if len(parameter) == 0:
        print("the parameter is empty")
    else:
        parameter = parameter[0]
        depth_range, depth_error_rate, best_depth = check_depth(parameter)
        min_data_records_list, data_records_error_rate, best_min_data_records = check_data_records(parameter, best_depth)
        node_purity_list, node_purity_error_rate, best_node_purity = check_node_purity(parameter, best_depth, best_min_data_records)
        print(depth_range, depth_error_rate, best_depth)
        print(min_data_records_list, data_records_error_rate, best_min_data_records)
        print(node_purity_list, node_purity_error_rate, best_node_purity)
