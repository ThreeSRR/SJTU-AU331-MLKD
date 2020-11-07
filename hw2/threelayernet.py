import numpy as np

class ThreeLayerNet(object):
    
    def __init__(self, input_size, hidden_size, output_size, std=1e-3):

        """ Three layer net implemented with numpy
        Weights and biases are stored in the variable self.params, which is a dictionary with the following keys:
        W1: Hiden layer weights, shape (D, H)
        b1: Hiden layer biases, shape (H,)
        W2: Second layer weights, shape (H, C)
        b2: Second layer biases, shape (C,)

        :param input_size: The dimension D of the input data.
        :param hidden_size: The number of neurons H in the hidden layer.
        :param output_size: The number of classes C.

        """
        self.output_size = output_size
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_sigmoid(self, x):
        return np.dot(self.sigmoid(x).T, (1 - self.sigmoid(x)))

    def MSELoss(self, predict, ground_truth):
        return np.sum(0.5*(predict-ground_truth)**2)
    
    def softmax(self, x):
        exp_x = np.exp(x)
        tmp = np.sum(exp_x, axis=1).reshape(x.shape[0], 1)
        return exp_x / tmp

    def loss(self, X, y):
        """Compute the loss and gradients, combining forward and backward
        :param X: training samples, shape (N, D)
        :param y: one-hot-coded training labels, shape (N，C)

        Returns:
        loss: Loss for this batch of training samples.
        grads: Dictionary mapping parameter names to gradients of those parameters

        """
        
        # Forward
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        a1 = X
        z1 = np.dot(a1, W1) + b1
        a2 = self.sigmoid(z1)
        z2 = np.dot(a2, W2) + b2

        loss = self.MSELoss(z2, y)

        # Backward pass: compute gradients
        grads = {}
        grads['W2'] = np.dot(a2.T, z2 - y)
        grads['b2'] = np.sum(z2 - y, axis=0)

        da2 = np.dot((z2 - y), W2.T)
        dz1 = np.dot(da2, self.derivative_sigmoid(z1))
        grads['W1'] = np.dot(a1.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return loss, grads

    def train(self, X, y, learning_rate=1e-3, batch_size=3):
        """Train the neural network
        :param X: training samples, shape (N, D)
        :param y: one-hot-coded training labels, shape (N, C)
        :param learning_rate: Scalar giving learning rate for optimization.
        :param batch_size: Number of training examples to use per step.
        
        Returns: loss and train_acc

        """

        num_train = X.shape[0]
        batch_num = max(num_train // batch_size, 1)

        total_loss = 0

        for idx in range(batch_num):

            X_batch = X[idx * batch_size : (idx+1) * batch_size]
            y_batch = y[idx * batch_size : (idx+1) * batch_size]

            loss, grads = self.loss(X_batch, y_batch)
            total_loss += loss

            # Gradient Descent

            dW1, db1 = grads['W1'], grads['b1']
            dW2, db2 = grads['W2'], grads['b2']
            
            self.params['W1'] -= learning_rate * dW1
            self.params['b1'] -= learning_rate * db1
            self.params['W2'] -= learning_rate * dW2
            self.params['b2'] -= learning_rate * db2

        # Check accuracy
        _, y_pred = self.predict(X)
        y_true = y
        
        count = 0
        for i in range(y_pred.shape[0]):
            if all(y_pred[i] == y_true[i]):
                count += 1
        train_acc = count / num_train
        loss = total_loss / num_train

        return {
          'loss': loss,
          'train_acc': train_acc,
        }
    
    def test(self, X, y):
        """Test the neural network
        :param X: testing samples, shape (N, D)
        :param y: one-hot-coded testing labels, shape (N, C)
        
        Returns: loss and test_acc

        """
        num_test = X.shape[0]

        # Check accuracy
        output, y_pred = self.predict(X)
        y_true = y
        
        count = 0
        for i in range(y_pred.shape[0]):
            if all(y_pred[i] == y_true[i]):
                count += 1
        test_acc = count / num_test
        loss = self.MSELoss(output, y_true) / num_test

        return {
          'loss': loss,
          'test_acc': test_acc,
        }


    def predict(self, X):

        """predict label of sample X
        :param X: testing data, shape (N, D)

        Returns:
        pred: predicted one-hot coded labels, shape (N, C)

        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        a1 = X
        z1 = np.dot(a1, W1) + b1
        a2 = self.sigmoid(z1)
        z2 = np.dot(a2, W2) + b2
        y_pred = np.argmax(z2, axis=1)
        
        # one-hot
        pred = np.zeros([y_pred.shape[0],self.output_size])
        for i in range(y_pred.shape[0]):
            pred[i][int(y_pred[i])] = 1

        return z2, pred
