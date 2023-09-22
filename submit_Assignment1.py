import numpy as np


# You are not allowed to use any ML libraries e.g. sklearn, scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SKLEARN, SCIPY, KERAS,TENSORFLOW ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHOD my_fit BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length
# def  ST(x,lamda):
#     return np.sign(x)*np.maximum(np.abs(x)-lamda,0)


def HT(v, k):
    t = np.zeros_like(v)
    if k < 1:
        return t
    else:
        ind = np.argsort(abs(v))[-k:]
        t[ind] = v[ind]
        return t


def predict(X_trn, model):
    return X_trn.dot(model)


def calculate_mae(X_trn, y_trn, model):
    predictions = predict(X_trn, model)
    mae = np.mean(np.abs(predictions - y_trn))
    return mae


################################
# Non Editable Region Starting #
################################
def my_fit(X_trn, y_trn):
    ################################
    #  Non Editable Region Ending  #
    ################################
    # #  updating w using coordinate descent
    # N,D=X_trn.shape  # N=no.of data samples ,D=no.of features
    # model= np.zeros(D) # initialise w to zero
    # w_prev=model.copy()  # soring the prev values of w
    # lamda=1
    # tolerance=1e-4
    # maxiter=100
    iter = 0
    # while(iter<maxiter):
    #     for j in range(D):
    #         r=y_trn-np.dot(X_trn,model) # residual
    #         # gradient of loss function ,which gives the form a*w-c
    #         # gradient of L1 norm is lamda*sign(w)
    #         # for optimimum of w , w= c+laamda*sign(w)/a
    #         a= np.dot(X_trn[:, j],X_trn[:, j]) # sum of all x*2
    #         c=r+model[j]*X_trn[:,j]  # resudal without w_j
    #         model[j]=ST(np.dot(X_trn[:,j],c),N*lamda)/a   #updating each wj
    #         if (np.allclose(w_prev, model, atol=tolerance)):
    #             break     #break condition

    #     w_prev=model.copy()
    #     iter=iter+1
    # Projection gradient method

    N, D = X_trn.shape  # N=no.of data samples ,D=no.of features
    model = np.zeros(D)  # initialise w to zero
    neta = 0.001  # learning rate (reduced further)
    # lamda=1 #thresold
    tolerance = 1e-4
    maxiter = 1000
    S = 512
    while (iter < maxiter):
        # calculate gradient of loss function
        gradient = (1 / N) * (X_trn.T.dot(X_trn.dot(model) - y_trn))
        model_new = model - neta * gradient
        # # apply soft thresholding
        # model=ST(model_new,lamda)
        # apply hard thresholding
        # Apply hard thresholding to keep the iterates sparse
        model = HT(model_new, S)
        # apply convergence
        if np.linalg.norm(model - model_new) < tolerance:
            break
        model = model_new
        iter = iter + 1

    # Use this method to train your model using training CRPs
    # Your method should return a 2048-dimensional vector that is 512-sparse
    # No bias term allowed -- return just a single 2048-dim vector as output
    # If the vector you return is not 512-sparse, it will be sparsified using hard-thresholding
    return model  # Return the trained model


X = np.loadtxt('dummy_test_challenges.dat')
Y = np.loadtxt('dummy_test_responses.dat')
w = my_fit(X, Y)

# Calculate the mean absolute error
mae = calculate_mae(X, Y, w)
print("Mean Absolute Error (MAE):", mae)

import time

start_time = time.time()
w = my_fit(X, Y)
end_time = time.time()
train_time = end_time - start_time

print("Training Time:", train_time, "seconds")