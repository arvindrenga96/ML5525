import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix
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


def rbf_svm_fit(X_train, y_train, c,gamma):
    """
    Train the model and return weight and bias
    """
    n_r, n_c = X_train.shape;
    K = rbf_kernel_matrix(X_train,X_train,gamma)
    P = matrix(np.outer(y_train, y_train) * K)
    q = matrix(np.ones(n_r) * -1)
    A = matrix(y_train, (1, n_r))
    b = matrix(np.zeros(1))
    G = matrix(np.vstack((np.eye(n_r) * -1, np.eye(n_r))));
    h = matrix(np.hstack((np.zeros(n_r), np.ones(n_r) * c)));
    solvers.options['show_progress'] = False
    Solution = solvers.qp(P, q, G, h, A, b);

    lagrangian = Solution['x']
    # Find support vectors
    X_sv = []
    y_sv = []
    lag_sv = []
    for i in range(len(lagrangian)):
        if lagrangian[i] > 10 ** -6:
            lag_sv.append(lagrangian[i])
            X_sv.append(X_train[i])
            y_sv.append(y_train[i])
    # Solving for weight w
    y_sv = np.array(y_sv).reshape(-1, 1)
    X_sv = np.array(X_sv)
    w = np.zeros((n_c, 1))
    for i in range(len(lag_sv)):
        w = np.add(w, lag_sv[i] * y_sv[i] * X_sv[i].reshape(n_c, 1))
    #find intercept
    model_intercept=y_sv[lag_sv.index(min(lag_sv))]
    for i in range(len(lag_sv)):
        model_intercept -= lag_sv[i] * y_sv[i] * rbf_kernel_matrix(X_sv[i], X_sv[lag_sv.index(min(lag_sv))],gamma)
    lag_sv = np.array(lag_sv).reshape(-1, 1)
    return lag_sv,X_sv,y_sv, model_intercept


def rbf_svm_predict(X_test, lag_value,X_supportvector,y_supportvector, model_intercept,gamma):
    pred_list = []  # initializing list where pred_y values will be stored
    #print(model_weights.shape)
    #print(y_supportvector.shape)
    for x_train in X_test:
        prediction = 0
        for i in range(len(lag_value)):
            prediction += lag_value[i] * y_supportvector[i] * rbf_kernel_matrix(X_supportvector[i], x_train,gamma)
        prediction += model_intercept
        #print(prediction)
        if (prediction>0):
            #print("Inside if")
            pred_value = 1
        else:
            pred_value = -1
            # append pred value to list of all pred value which will be used for
            # accurace computation
        pred_list.append(pred_value)
    return pred_list  # Return list of predicted values

def find_accuracy(pred_list, y_valid):
    """
    Find Accuracy and Confusion matrix values
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(pred_list)):  # loop through all predicted values
        pred_value = pred_list[i]
        # To judge whether predicted value belongs to 0 or 1 as it is logistic
        # regression
        if (pred_value == 1 and y_valid[i][0] >=0):  # for counting True Positives
            tp += 1
        if (pred_value == 1 and y_valid[i][0] <=0):  # for counting False postive
            fp += 1
        if (pred_value == -1 and y_valid[i][0] <= 0):  # for counting true negative
            tn += 1
        if (pred_value == -1 and y_valid[i][0] >=
                0):  # for counting False negative
            fn += 1
    return tp, fp, tn, fn

def rbf_kernel_matrix(X,Y,gamma):
    """
    Calculate and return the kernel matrix
    """
    if len(X)==len(Y):
        return np.exp(-gamma*np.linalg.norm(X-Y)**2)
    else:
        return [[np.exp(-gamma * (np.linalg.norm(X[i] - Y[j]))**2) for j in range(len(Y))] for i in range(len(X))]

if __name__ == '__main__':
    path = "hw2data.csv"
    data = pd.read_csv(path, header=None).to_numpy() ;
    np.random.seed(1)
    np.random.shuffle(data)
    X = data[:, 0:2]
    y = data[:, 2:3]
    bound = int(0.8*len(y))
    X_for_train = X[:bound]
    y_for_train = y[:bound]
    X_test = X[bound:]
    y_test = y[bound:]
    no_of_folds=10
    np.random.seed(1)
    X_Shuffled, y_shuffled = cross_Validation(
        X_for_train, y_for_train, no_of_folds)  # DO cross validation

    #C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    C=[1]
    #sigma = [0.1,0.25,0.5,1]
    sigma = [1]
    error_train = [];
    error_test = [];
    error_cv = [];
    for k in range(len(sigma)): #for varying sigma values
        for j in range(len(C)):
            train_error = 0
            validation_error = 0
            test_error=0
            for i in range(no_of_folds):  # looping through all k values of k fold to train and validated them individually
                # get value of X_train, y_train, X_valid, y_valid for present valid
                # fold
                X_Shuffled_train, y_shuffled_train, X_valid, y_valid = get_next_train_valid(
                X_Shuffled, y_shuffled,i)
                model_weights,x_supportvector,y_supportvector, model_intercept = rbf_svm_fit(
                X_Shuffled_train, y_shuffled_train,C[j],sigma[k])  # train the model for X_train, y_train
                #print(model_weights)
                #print(model_intercept)
                # Find predicted values for all rows in validation fold given
                # model_weights, model_intercept
                pred_list = rbf_svm_predict(X_valid,model_weights,x_supportvector,y_supportvector, model_intercept, sigma[k])
                # Find prediction values for training fold to be used for computing
                # training error
                #print(X_Shuffled_train)
                pred_list_training = rbf_svm_predict(X_Shuffled_train, model_weights,x_supportvector,y_supportvector, model_intercept, sigma[k])
                #print(pred_list_training)
                tp, fp, tn, fn = find_accuracy(
                pred_list_training, y_shuffled_train)  # tp,fp,tn,fn for training fold
                # computing train error rate
                train_error_rate = 1 - (tp + tn) / (tp + tn + fn + fp)
                #print("Train error for fold", i + 1, train_error_rate)
                # tp,fp,tn,fn for validation fold to be used for computing validation
                # error
                #print(y_valid)
                #print(pred_list)
                tp, fp, tn, fn = find_accuracy(pred_list, y_valid)
                # Computing Validation Eroor
                validation_error_rate = 1 - (tp + tn) / (tp + tn + fn + fp)
                # print("Validation error for fold", i + 1, validation_error_rate)

                #print("Confusion Matrix for fold", i + 1)  # Priniting Confusin Matrix
                #print(tp, fn)
                #print(fp, tn)
                # append train error rate to train_error_fold to be used for plotting
                pred_list = rbf_svm_predict(X_valid, model_weights,x_supportvector,y_supportvector, model_intercept, sigma[k])
                # Find prediction values for training fold to be used for computing
                # training error
                pred_list_testing = rbf_svm_predict(X_test, model_weights,x_supportvector,y_supportvector, model_intercept, sigma[k])
                tp, fp, tn, fn = find_accuracy(
                pred_list_testing, y_test)  # tp,fp,tn,fn for training fold
                # computing train error rate
                test_error_rate = 1 - (tp + tn) / (tp + tn + fn + fp)
                train_error+=train_error_rate
                # append validation error rate to validation_error_fold to be used for
                # plotting
                validation_error+=validation_error_rate
                test_error+=test_error_rate

            # printing all validation error rates for 5 folds
            print("Train Accuracy",C[j], 1-train_error/10)
            # prinitng all train_error_rates for all 5 folds
            print("validation Accuracy",C[j], 1-validation_error/10)
            print("TEST Accuracy",C[j], 1-test_error/10)
            error_train.append(1-train_error/10);
            error_cv.append(1-validation_error/10);
            error_test.append(1-test_error/10);

        plt.plot(C,error_test , label="Test")
        plt.plot(C, error_train, label="Train")
        plt.plot(C, error_cv, label="CV")
        plt.title("Accuracy-C graph Kernel SVM ")
        plt.xscale('log')
        plt.xlabel("C")
        plt.ylabel("Accuracy")
        plt.legend()

        #code For heatmap
            #Test_heat_map[k][j]=1-test_error*10
            #print(test_error)
        #fig, ax = plt.subplots()
        #im = ax.imshow(Test_heat_map)

        # We want to show all ticks...
        #ax.set_xticks(np.arange(len(4)))
        #ax.set_yticks(np.arange(len(4)))
        # ... and label them with the respective list entries
        #ax.set_xticklabels(sigma_arr)
        #ax.set_yticklabels(C)

        # Rotate the tick labels and set their alignment.
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         #        rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        #for i in range(4):
        #    for j in range(4):
         #       text = ax.text(j, i, Test_heat_map[i, j],
          #                     ha="center", va="center", color="w")

        #ax.set_title("Heat map and accuracy for hyperparameters")
        #fig.tight_layout()
        #plt.xlabel("sigma")
        #plt.ylabel("c")
        #plt.show()
