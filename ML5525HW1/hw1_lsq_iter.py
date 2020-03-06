import numpy as np
import matplotlib.pyplot as plt

# The following function computes closed form solution for least sqare
# given A and B


def lsq(A, B):
    num_columns = A.shape[1]  # getting no of coloumns
    rank = num_columns      # as all columns are independent so rank will be same
    # Singular value Decomposition
    U, sigma, VT = np.linalg.svd(A, full_matrices=False)
    # Calculating D^+
    D_plus = np.diag(np.hstack([1 / sigma, np.zeros(0)]))
    # calculating pseudo_inverse for A
    A_plus = VT.T.dot(D_plus).dot(U.T)
    # calcuating weights from pseudoinverse and B
    w = A_plus.dot(B)
    return w

# The following function computes iterative solution for least sqare
# given A and B


def lsq_iter(A, B):
    w_hat = lsq(A, B)  # calling closed form function for getting w_hat value
    mu = 1 / np.linalg.norm(A)  # Assigning value to mu
    n_epoch = 500  # setting epoch value
    # initialinzing w value as all zeroes as given in problem statement
    w_old = np.zeros(len(A[0]))
    all_errors = []  # Creating a list for storing all norm error values
    # Iteratate till n_epoch times to find converging w value
    for epoch in range(n_epoch):
        # using equation as given in book and w_old = W^k and w=W^(k+1)
        w = w_old - mu * np.matmul(A.T, np.matmul(A, w_old) - B)
        print("epoch", epoch + 1)
        # norm of difference between present w ie w^(k+1) and w_hat from closed
        # form solution
        error = np.linalg.norm(w - w_hat)
        w_old = w[:]  # assigning w to w_old for next iteration
        print("error", error)
        # append norm error value to all errors which will be used for plotting
        all_errors.append(error)
    return w_old, all_errors


if __name__ == '__main__':
    # initializing randomly an array of shape(20,10)
    A = 0.1 * np.random.rand(20, 10)
    # initializing randomly an array of shape(20,1)
    B = 10 * np.random.rand(20)
    w_hat, all_errors = lsq_iter(A, B)  # Calling Iterative Lsq solver
    epoch_values = []
    for epoch in range(
            500):  # For accumulating epoch values together for the plot
        epoch_values.append(epoch + 1)
    plt.figure(figsize=[20, 15])  # figure size
    # Plot epoch value vs all_errors
    plt.plot(epoch_values, all_errors, linewidth=3.0)
    # setting label for graph plot line
    plt.legend(['Error_Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)  # setting x axis label
    plt.ylabel('Loss', fontsize=16)  # setting y axis label
    plt.title('Error_Loss(norm) Vs epoch for showing convergence',
              fontsize=16)  # setting title of graph
    plt.show()  # show plot
