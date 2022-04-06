import numpy as np
import pandas as pd
import matplotlib

""" 
You are allowed to change the names of function "arguments" as per your convenience, 
but it should be meaningful.

E.g. y, y_train, y_test, output_var, target, output_label, ... are acceptable
but abc, a, b, etc. are not.
"""

def gradient(phi,w,y,n):
    gradient = [0]*n
    y = pd.DataFrame(y).to_numpy()
    diff = np.transpose(np.subtract(np.dot(phi,w),y))
    for i in range(0,n):
        gradient[i] = np.dot(diff,np.transpose(np.transpose(phi)[i]))
        gradient[i] = gradient[i]*(2/n)
    gradient = pd.DataFrame(gradient).to_numpy()
    return gradient

def p_norm_gradient(phi,w,y,lambada,p,n):
    gradient = [0]*n
    y = pd.DataFrame(y).to_numpy()
    diff = np.transpose(np.subtract(np.dot(phi,w),y))
    for i in range(0,n):
        gradient[i] = np.dot(diff,np.transpose(np.transpose(phi)[i]))
        gradient[i] = gradient[i]*(2/n)
        gradient[i] += lambada*p*(abs(np.transpose(w)[0][i]))**(p-1)
    gradient = pd.DataFrame(gradient).to_numpy()
    return gradient

def get_labeled_features(file_path):
    """Read data from train.csv and split into train and dev sets. Do any
       preprocessing/augmentation steps here and return final features.
    
    Args:
        file_path (str): path to train.csv

    Returns:
        phi_train, y_train, phi_dev, y_dev
    """
    df = pd.read_csv('train.csv')
    df = df.drop_duplicates()
    df['type'] = df['type'].replace(['white','red'],[0,1])
    test_ratio = 0.2
    train_ratio = 1- test_ratio
    train_count = int(train_ratio*len(df))
    train_X = df.iloc[:train_count,:-1].to_numpy()
    train_y = df.iloc[:train_count,-1].to_numpy()
    validate_X = df.iloc[train_count:,:-1].to_numpy()
    validate_y = df.iloc[train_count:,-1].to_numpy()
    
    return train_X,train_y, validate_X, validate_y

def get_test_features(file_path):
    """Read test data, perform required preproccessing / augmentation
       and return final feature matrix.

    Args:
        file_path (str): path to test.csv

    Returns:
        phi_test: matrix of size (m,n) where m is number of test instances
                  and n is the dimension of the feature space.
    """
    df = pd.read_csv('test.csv')
    df['type'] = df['type'].replace(['white','red'],[0,1])
    phi_test = df.values
    
    return phi_test

def compute_RMSE(phi, w , y) :
    """Return root mean squared error given features phi, and true labels y."""
    w = pd.DataFrame(w).to_numpy()
    phi = pd.DataFrame(phi).to_numpy()
    y = pd.DataFrame(y).to_numpy()
    phiW = np.dot(phi,w)
    diff = np.subtract(phiW,y)
    diff = np.transpose(diff)
    diff = diff[0]
    val = 0
    n = len(diff)
    for value in diff:
        val += (value**2)/n
    error = np.sqrt(val)
    return error

def generate_output(phi_test, w):
    """writes a file (output.csv) containing target variables in required format for Submission."""

    result = np.dot(phi_test,w)
    result = np.transpose(result)
    result = result[0]
    result = [int(val) for val in result]
    ids = [i for i in range(0,len(phi_test))]
    dict = {"id":ids,"quality":result}
    df = pd.DataFrame(dict)
    df.to_csv('output.csv',header=True,index=False)

    
   
def closed_soln(phi, y):
    """Function returns the solution w for Xw = y."""
    return np.linalg.pinv(phi).dot(y)
   
def gradient_descent(phi, y, phi_dev, y_dev) :
    # Implement gradient_descent using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    learning_rate = 0.000000001
    n = len(phi[0])
    w_0 = pd.DataFrame([0]*n).to_numpy()
    for i in range(0,5000):
        w_1 = np.subtract(w_0,np.multiply(learning_rate,gradient(phi,w_0,y,n)))
        w_0 = w_1
    return w_1

def sgd(phi, y, phi_dev, y_dev) :
    # Implement stochastic gradient_descent using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    learning_rate = 0.000000001
    data_count = len(phi)
    n = len(phi[0])
    w_0 = pd.DataFrame([0]*n).to_numpy()
    for i in range(0,5000):
        random_indices = np.random.randint(data_count,size = 100)
        new_phi =[]
        new_y =[]
        for i in random_indices:
            new_phi.append(phi[i])
            new_y.append(y[i])
        
        w_1 = np.subtract(w_0,np.multiply(learning_rate,gradient(new_phi,w_0,new_y,n)))
        w_0 = w_1
    return w_1


def pnorm(phi, y, phi_dev, y_dev, p) :
    # Implement gradient_descent with p-norm regularization using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    learning_rate = 0.000000001
    n = len(phi[0])
    w_0 = pd.DataFrame([0]*n).to_numpy()
    lambada = 1000
    for i in range(0,5000):
        w_1 = np.subtract(w_0,np.multiply(learning_rate,p_norm_gradient(phi,w_0,y,lambada,p,n)))
        w_0 = w_1
    return w_1
 

def main():
    """ 
    The following steps will be run in sequence by the autograder.
    """
    ######## Task 2 #########
    phi, y, phi_dev, y_dev = get_labeled_features('train.csv')
    w1 = closed_soln(phi, y)
    w2 = gradient_descent(phi, y, phi_dev, y_dev)
    r1 = compute_RMSE(phi_dev, w1, y_dev)
    r2 = compute_RMSE(phi_dev, w2, y_dev)
    print('1a: ')
    print(abs(r1-r2))
    w3 = sgd(phi, y, phi_dev, y_dev)
    r3 = compute_RMSE(phi_dev, w3, y_dev)
    print('1c: ')
    print(abs(r2-r3))

    ######## Task 3 #########
    w_p2 = pnorm(phi, y, phi_dev, y_dev, 2)  
    w_p4 = pnorm(phi, y, phi_dev, y_dev, 4)  
    r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
    r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
    print('2: pnorm2')
    print(r_p2)
    print('2: pnorm4')
    print(r_p4)

    ######## Task 6 #########
    
    # Add code to run your selected method here
    # print RMSE on dev set with this method
    test = get_test_features('test.csv')
    test = pd.DataFrame(test).to_numpy()
    generate_output(test,w2)
    
main()