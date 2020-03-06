import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def softmax(z):
    z -= np.max(z).T
    softmax = np.exp(z).T / np.sum(np.exp(z)).T
    return softmax.T
def cross_entropy(yhat, y):
    return - np.sum(yhat * np.log(y+1e-6))
def create_mini_batches(train_x,train_y, batch_size):
    no_of_batches = int(train_x.shape[0] / batch_size)  # Size of each fold
    # remainder value if any after k folds of particular size each are used
    rem = int(len(train_x)) % batch_size
    mini_batches=[]
    for i in range(no_of_batches):
        x_batch = train_x[batch_size * i:batch_size * (i + 1)]
        y_batch = train_y[batch_size * i:batch_size * (i + 1)]
        mini_batches.append((x_batch, y_batch))
    if(rem!=0):
        x_batch = train_x[batch_size * no_of_batches:len(train_x)]
        y_batch = train_y[batch_size * no_of_batches:len(train_x)]
        mini_batches.append((x_batch, y_batch))
    return mini_batches
def mini_batch_gradient(W, b, X_mini_train, y_mini_train):
    batch_size = X_mini_train.shape[0]

    n = len(y_mini_train)
    y = np.zeros((n, 10))
    for i in range(n):
        y[i,int(y_mini_train[i])] = 1
    value = np.dot(X_mini_train[i], W)
    # change value into probability with softmax
    prob = softmax(value)
    # loss function
    loss = 1/(cross_entropy(y,prob)/batch_size)
    #print("loss",loss)
    # gradient of loss function
    grad = (-1 / batch_size) * np.dot(X_mini_train.T, (y - prob))
    #print("grad",grad)
    return loss, grad

def multi_logistic_train(train_x,train_y,batch_size, learning_rate, n_epochs=1):
    """
    Train the model using mini-batch gradient descent and return the model weight
    """

    d = train_x.shape[1]
    k = len(np.unique(train_y))
    #print(train_y)
    # initialize w
    W = np.random.rand(d, 10)*0.01
    b = np.random.randn(10, 1) * 0.01
    loss_list = []
    batch_no = []
    batch_num=0
    mini_batches = create_mini_batches(train_x,train_y, batch_size)
    for i in range(n_epochs):
        for mini_batch in mini_batches:
            batch_num += 1
            batch_no.append(batch_num)
            X_mini_train, y_mini_train = mini_batch
            #print(y_mini_train)
            #print("Xmini",X_mini_train)
            batch_loss, grad = mini_batch_gradient(W, b, X_mini_train, y_mini_train)
            W = W - learning_rate * grad
            loss_list.append(batch_loss)

    return W, loss_list, batch_no
def multi_logistic_predict(X, w):
    """
    predict labels of input data using
    """
    prob = softmax(np.dot(X,w))
    # find the largest probability, take this class as label of input data
    y_predict = np.argmax(prob, axis=1)
    return y_predict
def find_accuracy(y_predict,y):
    """
    Predict the accuracy given real y and predicted y and also calculate confusion matrix
    """
    confusion_matrix = np.zeros((10, 10))
    accuracy=0
    for i in range(len(y)):
        if(y_predict[i] == y[i]):
            accuracy+=1
    accuracy = accuracy/float(len(y))
    for i in range(len(y)):
      confusion_matrix[y[i]][y_predict[i]] =  confusion_matrix[y[i]][y_predict[i]] + 1
    return accuracy,confusion_matrix

if __name__ == '__main__':
    mnist_train = pd.read_csv('mnist_train.csv').values;
    mnist_test = pd.read_csv('mnist_test.csv').values ;
    np.random.seed(0)
    print(mnist_train.shape)
    train_y= mnist_train[:,0]
    test_y = mnist_test[:,0]
    train_x =mnist_train[:,1:]
    test_x =mnist_test[:,1:]
    #normalizing trainx and train y
    train_x= (train_x- train_x.mean())/train_x.std()
    test_x = (test_x-test_x.mean())/test_x.std()
    #Initial weight and bias
    #n_epochs = 15
    batch_size = 32
    learning_rate = 0.01
    W, loss_list, num_of_batches = multi_logistic_train(train_x,train_y,batch_size, learning_rate)
    print(len(W))
    print(len(num_of_batches))
    print(loss_list)
    np.savetxt('weight_multi_logistic.out',W)
    y_predict = multi_logistic_predict(test_x, W)
    accuracy,confusion_matrix = find_accuracy(y_predict,test_y)
    print(accuracy)
    print(confusion_matrix.astype(int))
    plt.plot(num_of_batches,loss_list)
    plt.title("loss vs batch number")
    plt.xlabel("batch number")
    plt.ylabel("loss")
    plt.show()



