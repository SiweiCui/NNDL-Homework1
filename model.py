import numpy as np

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_backward(x): s = sigmoid(x); return s * (1 - s)

def relu(x): return np.maximum(0, x)
def relu_backward(x): return (x > 0).astype(float)

activation_functions = {
    "relu": (relu, relu_backward),
    "sigmoid": (sigmoid, sigmoid_backward),
}

class Model:
    def __init__(self, input_size, hidden_size, output_size, activation="relu", l2_lambda=0.0, lr = 0.01):
        # 模型结构相关
        self.hidden_size = hidden_size
        self.activation, self.activation_deriv = activation_functions[activation]
        # 正则化参数
        self.l2_lambda = l2_lambda
        self.lr = lr
        # 权重, 偏置及其梯度
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.dW1 = np.zeros_like(self.W1)
        
        self.b1 = np.zeros((1, hidden_size))
        self.db1 = np.zeros_like(self.b1)

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.dW2 = np.zeros_like(self.W2)

        self.b2 = np.zeros((1, output_size))
        self.db2 = np.zeros_like(self.b2)

    def forward(self, X):
        '''
        前向传播
        '''
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, X, y_true):
        '''
        反向传播
        '''
        m = y_true.shape[0]
        y_onehot = np.zeros_like(self.a2)
        y_onehot[np.arange(m), y_true] = 1

        dz2 = (self.a2 - y_onehot) / m
        self.dW2 = self.a1.T @ dz2 + self.l2_lambda * self.W2
        self.db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.activation_deriv(self.z1)
        self.dW1 = X.T @ dz1 + self.l2_lambda * self.W1
        self.db1 = np.sum(dz1, axis=0, keepdims=True)

    def update_params(self):
        '''
        更新参数
        '''
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2

    def cross_entropy_loss(self, y_pred, y_true):
        '''
        计算交叉熵损失, 用于画图
        '''
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        data_loss = np.sum(log_likelihood) / m
        reg_loss = 0.5 * self.l2_lambda * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return data_loss + reg_loss

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path):
        data = np.load(path)
        self.W1, self.b1 = data['W1'], data['b1']
        self.W2, self.b2 = data['W2'], data['b2']