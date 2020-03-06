import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")


# Compute Sigmoid value Given w and x
def sigmoid(w, x):
    sig = (1 / (1 + np.exp(-np.dot(w.T, x))))
    print(sig)
    return sig


# For training Logistic regression model using Gradient Descent
def train(X, y):
    l_rate = 0.1  # Learning rate
    n_epoch = 2000  # max number of epochs it will run if doesnt converge before that
    converge_change = 0.002  # Setting an initial value for convergence
    # append y value to X to make a single augmented matrix
    X = np.append(X, y, 1)
    # initializing all value as randomly for w
    w = [np.random.rand(1)[0] for i in range(len(X[0]))]
    w_old = w[:]  # initializing w_old also to the value of w
    # The below loop is where gradient descent happens
    for epoch in range(n_epoch):
        sum_error = 0  # initalizing sum of all errors as 0
        for i in range(X.shape[0]):
            # calling predict_row with X[i] and w which computes y_hat value
            # given a row of values and weights vector
            y_hat = predict_row(X[i], w)
            # computing error which is calculated as difference between actual
            # value y in and computed value y_hat
            error = X[i][-1] - y_hat
            sum_error += error * error  # sum sqaure of errors
            # Computing intercept's new value by descent along the slope
            w[0] = w[0] + l_rate * error * y_hat * (1 - y_hat)
            for j in range(
                    len(X[i]) - 1):  # Looping through all the columns for to compute value of weights
                w[j + 1] = w[j + 1] + l_rate * error * \
                    y_hat * (1 - y_hat) * X[i][j]
        # Calculating change value for convergence condition
        change_w = np.linalg.norm(np.array(w_old) - np.array(w))
        w_old = w[:]  # just a deepcopy of w to w_old for next iteration
        if (change_w < converge_change):  # convergence criterion
            # print("Converged")
            break  # break if convergence criterion satisfied and stop looping through epochs
    return w[1:], w[0]  # return intercept and weights


# Function which computes y_hat value given a row of values or row vector
# and weights vector
def predict_row(row, w):
    yhat = w[0]
    for i in range(len(row) - 1):
        yhat += w[i + 1] * row[i]
    return 1.0 / (1.0 + np.exp(-yhat))  # Computing Sigmoid value


# Function to compute and return list of predicted value of y given
# x_test,model_weights and model_intercept
def predict(x_test, model_weights, model_intercept):
    pred_list = []  # initializing list where pred_y values will be stored
    for j in range(len(x_test)):
        yhat = model_intercept
        for i in range(
                len(x_test[j])):  # Looping through all values of a given row in X_test
            # Mutlplying X_test row's individual value with respective weights
            # to find yhat
            yhat += model_weights[i] * x_test[j][i]
        # Find use value from previous step to compute sigmoid value
        pred_value = 1.0 / (1.0 + np.exp(-yhat))
        # append pred value to list of all pred value which will be used for
        # accurace computation
        pred_list.append(pred_value)
    return pred_list  # Return list of predicted values

# randomly shuffles the dataset and partitions it into almost equal(k=5)
# folds.Returns dictionaries  Xshuffled  and yshuffled which is 5 folds
# stored into dictionaries


def cross_Validation(matrix, target, folds=5):
    matrix_split = list()
    matrix_split_results = list()
    matrix_copy = matrix[:]
    matrix_target_copy = target[:]
    fold_size = int(len(matrix) / folds)  # Size of each fold
    # remainder value if any after k folds of particular size each are used
    rem = int(len(matrix)) % folds
    for i in range(folds):  # For partitioning folds
        fold = list()
        foldtarget = list()
        if rem >= 1:  # if remainder would have been greater than 1 then we will have unequal fold size so an if statement to handle additional values to be partitioned
            while len(fold) < fold_size + \
                    1:  # to check if fold has reached its required size
                # randomly choosing an index
                index = int(np.random.rand(1) * (len(matrix_copy)))
                chosen = matrix_copy[index]  # X train Value at chosen index
                # delete that value from list of values to partitioned to avoid
                # repetation
                np.delete(matrix_copy, index)
                # same Thing for target as was done for x train
                chosen_target = matrix_target_copy[index]
                np.delete(matrix_target_copy, index)
                # append Chosen value to a fold which is created
                foldtarget.append(chosen_target)
                fold.append(chosen)
            # append fold computed previously to dictionary of folds
            matrix_split.append(fold)
            matrix_split_results.append(foldtarget)
            rem = rem - 1  # rem is deprecated as one of the extra values have been used for fold
        else:
            while len(fold) < fold_size:
                index = int(np.random.rand(1) * (len(matrix_copy)))
                last = matrix_copy[index]
                np.delete(matrix_copy, index)
                last_target = matrix_target_copy[index]
                np.delete(matrix_target_copy, index)
                foldtarget.append(last_target)
                fold.append(last)
            matrix_split.append(fold)
            matrix_split_results.append(foldtarget)
    # assign the value of dictionary matrix_split, matrix_split_results to
    # X_shuffled, Y_shuffled  which is to be returned
    X_shuffled, Y_shuffled = matrix_split, matrix_split_results
    # return  dictionary  X_shuffled  and y_shuffled.
    return X_shuffled, Y_shuffled

# Given dictionary  X_shuffled  and y_shuffled and itr number compute next
# train valid split and return X_train,  y_train,  X_valid,  y_valid


def get_next_train_valid(X_Shuffled, y_shuffled, itr):
    train_x = X_Shuffled.copy()
    train_y = y_shuffled.copy()
    test_x = X_Shuffled[itr]
    test_y = y_shuffled[itr]
    # remove validation fold from dictionary so that we can return the rest of
    # dictioary for train folds
    del train_y[itr]
    del train_x[itr]
    # Concatening all folds of train together which will be returned
    y_train = np.concatenate(train_y)
    X_train = np.concatenate(train_x)
    X_valid = test_x  # validation Fold of x values
    y_valid = test_y
    return X_train, y_train, X_valid, y_valid

# Given List of predicted value by a model and actual values Compute True
# Positive ,False positive,True negative and false Negative


def find_accuracy(pred_list, y_valid):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(pred_list)):  # loop through all predicted values
        pred_prob = pred_list[i]
        # To judge whether predicted value belongs to 0 or 1 as it is logistic
        # regression
        pred_value = np.where(pred_prob >= .5, 1, 0)
        if (pred_value == 1 and y_valid[i] ==
                1):  # for counting True Positives
            tp += 1
        if (pred_value == 1 and y_valid[i] == 0):  # for counting False postive
            fp += 1
        if (pred_value == 0 and y_valid[i] == 0):  # for counting true negative
            tn += 1
        if (pred_value == 0 and y_valid[i] ==
                1):  # for counting False negative
            fn += 1
    return tp, fp, tn, fn


if __name__ == '__main__':
    iris = pd.read_csv(
        "IRISFeat.csv",
        header=None).as_matrix()  # read IRIS data
    iris_target = pd.read_csv(
        "IRISlabel.csv",
        header=None).as_matrix()  # read iris data actual labels
    # setting random seed to avoid randomness in folds and answers
    np.random.seed(2)
    X_Shuffled, y_shuffled = cross_Validation(
        iris, iris_target, 5)  # DO cross validation
    train_error_folds = []
    validation_error_folds = []
    for i in range(
            5):  # looping through all k values of k fold to train and validated them individually
        # get value of X_train, y_train, X_valid, y_valid for present valid
        # fold
        X_train, y_train, X_valid, y_valid = get_next_train_valid(
            X_Shuffled, y_shuffled, i)
        model_weights, model_intercept = train(
            X_train, y_train)  # train the model for X_train, y_train

        # Find predicted values for all rows in validation fold given
        # model_weights, model_intercept
        pred_list = predict(X_valid, model_weights, model_intercept)
        # Find prediction values for training fold to be used for computing
        # training error
        pred_list_training = predict(X_train, model_weights, model_intercept)
        tp, fp, tn, fn = find_accuracy(
            pred_list_training, y_train)  # tp,fp,tn,fn for training fold
        # computing train error rate
        train_error_rate = 1 - (tp + tn) / (tp + tn + fn + fp)
        print("Train error for fold", i + 1, train_error_rate)
        # tp,fp,tn,fn for validation fold to be used for computing validation
        # error
        tp, fp, tn, fn = find_accuracy(pred_list, y_valid)
        # Computing Validation Eroor
        validation_error_rate = 1 - (tp + tn) / (tp + tn + fn + fp)
        print("Validation error for fold", i + 1, validation_error_rate)

        print("Confusion Matrix for fold", i + 1)  # Priniting Confusin Matrix
        print(tp, fn)
        print(fp, tn)
        # append train error rate to train_error_fold to be used for plotting
        train_error_folds.append(train_error_rate)
        # append validation error rate to validation_error_fold to be used for
        # plotting
        validation_error_folds.append(validation_error_rate)
    # printing all validation error rates for 5 folds
    print("Train error Folds", train_error_folds)
    # prinitng all train_error_rates for all 5 folds
    print("validation error folds", validation_error_folds)
    n_groups = 5  # No of groups for bar graph
    means_train = train_error_folds
    means_validation = validation_error_folds

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
    # defining respective rectangles and thier properties for bar graph
    rects1 = plt.bar(index, means_train, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Train Error')

    rects2 = plt.bar(index + bar_width, means_validation, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Validation Error')

    plt.xlabel('Validation Error Rate')  # x label for plot
    plt.ylabel('Folds')  # y label for plot
    plt.title('Scores by Fold')  # tile of the graph
    plt.xticks(
        index + bar_width,
        ('Fold 1',
         'Fold 2',
         'Fold 3',
         'Fold 4',
         'Fold 5'))  # X axis tickers

    plt.tight_layout()
    plt.show()  # show Plot
