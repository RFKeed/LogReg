from numpy.matrixlib.defmatrix import matrix
import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegression:
    ''' 
    Parameters:
    W: {matrix} -> initialization of the weights of the model, which will be updated in the future during the training process
    b: {float} -> Initialization of b, the number in which will change during the learning process when taking the derivative of the error for the best result. The purpose of b is to adjust the linear relationship to account for baselines in the response variable.
    loss: {matrix - float} -> initialization of an error that will change in the future during the learning process. The error was created to indicate the training of the model. How well does the model predict the correct targets and how much is it wrong when giving answers.


    Logistic regression is a fundamental classification method. It belongs to the group of linear classifiers and is somewhat similar to polynomial and linear regression. 
    Logistic regression is fast and relatively simple, and it is convenient for you to interpret the results. 
    Although this is essentially a binary classification method, it can also be applied to multiclass problems.

    Examples

    X = np.arange(10).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    log_reg = LogisticRegression().fit(X, y, 1000)
    log_reg.predict(X[:2, :])

    '''
    def __init__(self):
        """
        Inizializing model
        This magic is a function that creates specific attributes of a class. In our case, these are only hyperparameters of the neural network and the values of the error function (according to the standard, it is 0, since in the future this attribute will be reassigned)

        

        Example
        log_reg = LogisticRegression()
        """
        # variables for storing weights
        self.W, self.b = None, None
        # variable for storing current loss
        self.loss = None

    def accuracy(self, y: np.array, p:np.array) -> float:
        """
        Parameters:
        y: 1d array-like, or label indicator array / sparse matrix
        p: 1d array-like, or label indicator array / sparse matrix

        This function return accuracy score between our prediction and real classes

        Examples
        y_pred = [0, 2, 1, 3]
        y_true = [0, 1, 2, 3]
        accuracy_score(y_true, y_pred)
        """
        return 1 - accuracy_score(p, y)

    def cost_function(self, p: np.array, y: np.array) -> np.array:
        """
        Parameters:
        p: (array-like / matrix) predicted classes by model
        y: (array-like / matrix) real classes

        Cross-entropy is a Loss Function that can be used to quantify the difference between two Probability Distributions.
        """
    
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def _init_weights(self):
        """
        This function is designed for random initialization of weights for our neural network. we will reassign the attributes of our class
        W: random initialization of the weights of the model, which will be updated in the future during the training process
        b: random initialization of b, the number in which will change during the learning process when taking the derivative of the error for the best result. The purpose of b is to adjust the linear relationship to account for baselines in the response variable.
        """
        # initialize normal
        self.W = np.random.randn(3, 1)
        self.b = np.random.randn()

    def sigm(self, x:int):
        """
        Parameters:
        x: int, the value (number/matrix) by which the sigmoid graph is plotted by function to determine the class by the probability of the relation to it

        A sigmoid is a smooth monotonic increasing nonlinear function having the shape of the letter "S", which is often used to "smooth" the values of a certain value. It is used to obtain predictions of the logistic regression class. 

        x = 2
        sigm(2)
        """
        # sigmoid (logistic) function
        return 1 / (1 + np.exp(-x))

    def forward_backward_pass(self, x: np.array, y: np.array, eta: float):
        """
        Parameters:
        x: array / matrix data
        y: array / matrix target
        eta: (float) learning rate

        This function implements forward and backward pass and updates the parameters W / b. 
        First, we get linear predictions from our model by multiplying the matrix by the matrix of feature weights and adding b. After that, using sigmoids, we get targets for specific instances. 
        After calculating the error (cross-entropy), our forward pass is over. 
        During the reverse transfer, we calculate the derivatives of our error function with by W and b, and then update them taking into account the learning rate. 
        "passage forward and backward."
        """
        # FORWARD
        linear_pred = np.dot(x, self.W) + self.b
        y_pred = self.sigm(linear_pred)
        # FORWARD ENDS

        # calculate loss
        self.loss = self.cost_function(y_pred, y)

        # BACKWARD
        # here you need to calculate all the derivatives of loss with respect to W and b

        dLdW = (y_pred - y) * x.T
        dLdb = (y_pred - y)

        # then update W and b
        # don't forget the learning rate - 'eta'!
       
        self.W = self.W - eta * dLdW
        self.b = self.b - eta * dLdb

        # BACKWARD ENDS

    def fit(self, X: np.array, Y: np.array, eta=0.01, decay=0.999, iters=1000) ->list:
        """
        X: array of data
        Y: array of target
        eta: float, learning rate
        decay: float, updationg learning rate
        iters: num of iter


        This function is designed to train our logistic regression. First we initialize random weights. 
        Iteratively (depending on the number of epochs) we will perform forward-backward-pass, changing hyperparameters and learning rate. 
        We also add the error we calculated to the buffer for future graph output (as our model learns)

        return: We  get a list of our model's errors to plot the model's training schedule.

        Examples
        X = np.arange(10).reshape(-1, 1)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        log_reg = LogisticRegression()
        log_reg.fit(X, y, 1000)
        """
        self._init_weights()

        # buffer - for printing out mean loss over 100 iterations
        buffer = []
        # L - history of losses
        L = []

        # perform iterative gradient descent
        for i in range(iters):
            index = np.random.randint(0, len(X))
            x = X[index]
            y = Y[index]
            # update params
            self.forward_backward_pass(x, y, eta)
            # update learning rate
            eta *= decay

            L.append(self.loss)
            buffer.append(self.loss)

        return L

    def predict(self, x: np.array) -> np.array:
        """
        x: array of test-data

        The function is designed to predict the class of a particular instance based on its attributes. 
        return: the function returns the prediction of the class by the sigmoid of our model based on the hyperparameter

        Examples
        X = np.arange(10).reshape(-1, 1)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        log_reg = LogisticRegression().fit(X, y, 1000)
        log_reg.predict(X[:2, :])



        """
        # Note you have to return actual classes (not probs)
        linear_pred = np.dot(x, self.W) + self.b
        y_pred = self.sigm(linear_pred)
        return np.round(y_pred)
