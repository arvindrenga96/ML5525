import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from cvxopt import solvers, matrix
import hw2_multi_logistic
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
    return lag_sv,X_sv,y_sv, w,model_intercept

def mnist_svm_train(X_train, y_train, C,gamma):
    X = X_train
    model_lambda = []
    x_supportvector = []
    y_supportvector = []
    model_intercept = []
    weights = []
    weights_together =[]
    for c in range(0, 10):
        y_train_use = copy.deepcopy(y_train)
        np.place(y_train_use, y_train_use != c, -1)
        np.place(y_train_use, y_train_use == c, 1)  # replace the label 10 with 0
        # print(y_train_use)
        # print(X)
        lamda, x_sv, y_sv, w, intercept = rbf_svm_fit(X, y_train_use, C, gamma)
        model_lambda.append(lamda)
        x_supportvector.append(x_sv)
        y_supportvector.append(y_sv)
        model_intercept.append(intercept)
        weights.append(w)
    np.savetxt('weight_multi_svm.out',weights[:][:][0])
    return model_lambda,x_supportvector,y_supportvector,model_intercept,weights
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
        #print(model_intercept)
        if (prediction>0):
            #print("Inside if")
            pred_value = 1
        else:
            pred_value = -1
            # append pred value to list of all pred value which will be used for
            # accurace computation
        pred_list.append(pred_value)
    return pred_list  # Return list of predicted values

def find_accuracy(y_predict,y):
    """
    Predict the accuracy given real y and predicted y and also calculate confusion matrix
    """
    confusion_matrix = np.zeros((10, 10))
    count = 0
    #accuracy = sum(y_predict == y) / float(len(y))
    for i in range(len(y)):
        if(y_predict[i] == y[i]):
            count=count+1
    accuracy = count*100/len(y)
    for i in range(len(y)):
     # print("iteration",i)
      #print(y[i])
      #print(y_predict[i])
      confusion_matrix[int(y[i])][y_predict[i]] =  confusion_matrix[int(y[i])][y_predict[i]] + 1
    return accuracy,confusion_matrix

def rbf_kernel_matrix(X,Y,gamma):
    """
    Calculate and return the kernel matrix
    """
    if len(X)==len(Y):
        return np.exp(-gamma*np.linalg.norm(X-Y)**2)
    else:
        return [[np.exp(-gamma * (np.linalg.norm(X[i] - Y[j]))**2) for j in range(len(Y))] for i in range(len(X))]
def mnist_svm_predict(X_test, model_lambda, x_supportvector, y_supportvector, model_intercept, gamma=0.5):
    pred_list =[]
    for c in range(0, 10):
        pred=rbf_svm_predict(X_test, model_lambda[c], x_supportvector[c], y_supportvector[c], model_intercept[c], gamma)
        #print("c",c)
        #print(pred)
        pred_list.append(pred)
    return pred_list
def one_versus_all(pred_list):
    predlist_array = np.array(pred_list)
    #print(predlist_array.shape)
    result = np.argmax(predlist_array, axis=0)
    return result
def soft_max_regression(train_x,train_y,test_x,test_y):
    batch_size = 32
    learning_rate = 0.01
    W, loss_list, num_of_batches = hw2_multi_logistic.multi_logistic_train(train_x,train_y,batch_size, learning_rate)
    y_predict = hw2_multi_logistic.multi_logistic_predict(test_x, W)
    accuracy,confusion_matrix = find_accuracy(y_predict,test_y)
    return accuracy,confusion_matrix
if __name__ == '__main__':
    mnist_train = pd.read_csv('mfeat_train.csv').values;
    mnist_test = pd.read_csv('mfeat_test.csv').values;
    np.random.seed(0)
    print(mnist_train.shape)

    mnist_train = mnist_train[:, 1:]
    mnist_test = mnist_test[:, 1:]
    X_train = mnist_train[:, 0:-1]
    y_train = mnist_train[:,-1]
    #X_train= (X_train- X_train.mean())/X_train.std()
    #X_test = (X_test-X_test.mean())/X_test.std()
    X_test=mnist_test[:, 0:-1]
    y_test = mnist_test[:,-1]
    np.place(y_train, y_train == 10, 0)  # replace the label 10 with 0
    np.place(y_test, y_test == 10, 0)  # replace the label 10 with 0
    #print(X_train.shape)
    #print(y_train.shape)
    #print(X_test.shape)
    #print(y_test.shape)
    model_lambda,x_supportvector,y_supportvector,model_intercept,weights  = mnist_svm_train(X_train, y_train, C=1, gamma=0.5)
   # print(model_intercept)
    #print(y_train)
    #print(len(model_weights))
    #print(len(model_weights[0]))
    pred_list=mnist_svm_predict(X_test, model_lambda, x_supportvector, y_supportvector, model_intercept, gamma=0.5)
    #print(pred_list)

    #print(len(pred_list[0]))
    result = one_versus_all(pred_list)

    accuracy,confusion_matrix = find_accuracy(result,y_test)
    print(accuracy)
    print(confusion_matrix.astype(int))
    #print(model_intercept.shape)
    #print(weights.shape)
    accuracy_softmax,confusion_matrix_softmax=soft_max_regression(X_train, y_train, X_test, y_test)
    print(accuracy_softmax)
    print(confusion_matrix_softmax)

    """ As accuracy was too low checking with scikit learn i ended up getting similar results for rbf kernal scikit but result improved dramatically 
        for i in range(0,10):
        y_train_use = copy.deepcopy(y_train)
        np.place(y_train_use,y_train_use!=i ,-1) #all other than chosen class  -1 in one vs all
        np.place(y_train_use, y_train_use == i, 1)  #replace y for chosen class in one vs all as 1 meaning positive example
        #print(y_train_use)
        classifiers.append(SVC(kernel='rbf',gamma=0.5,C=1))
        #print(X)
        classifiers[i].fit(X, y_train_use)
    pred_list = []
    for i in range(0, 10):
        result =classifiers[i].predict(X_test)
        print("i",i,len(set(result)))
        #print(y_test)
        pred_list.append(result)
        predlist_array = np.array(pred_list)
    #print(predlist_array.shape)
    result = np.argmax(predlist_array, axis=0)
    accuracy,confusion_matrix = find_accuracy(result,y_test)
    print(accuracy)
    print(confusion_matrix.astype(int))    
        """
