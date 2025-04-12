import argparse
import numpy as np
import pickle
import os

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_backward(x): s = sigmoid(x); return s * (1 - s)

def relu(x): return np.maximum(0, x)
def relu_backward(x): return (x > 0).astype(float)

activation_functions = {
    "relu": (relu, relu_backward),
    "sigmoid": (sigmoid, sigmoid_backward),
}

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


# 必须与训练代码保持完全一致的模型定义
class Model:
    def __init__(self, input_size, hidden_size, output_size, activation="relu"):
        self.activation, _ = self.get_activation(activation)
        # 初始化权重（实际权重由加载的npz文件覆盖）
        self.W1 = np.zeros((input_size, hidden_size))  # 占位符
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.zeros((hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    @staticmethod
    def get_activation(activation):
        return activation_functions[activation]

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.activation(z1)
        z2 = a1 @ self.W2 + self.b2
        return softmax(z2)

    def load(self, path):
        data = np.load(path)
        self.W1, self.b1 = data['W1'], data['b1']
        self.W2, self.b2 = data['W2'], data['b2']


def load_test_data(data_path):
    """专门加载测试集"""
    test_path = os.path.join(data_path, 'test_batch')
    with open(test_path, 'rb') as f:
        test_batch = pickle.load(f, encoding='latin1')
        test_data = test_batch['data'].astype(np.float32) / 255.0
        test_labels = np.array(test_batch['labels'])
    return test_data, test_labels

def compute_accuracy(model, X, y, batch_size=128):
    correct = 0
    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        y_pred = model.forward(X_batch)
        preds = np.argmax(y_pred, axis=1)
        correct += np.sum(preds == y_batch)
    return correct / X.shape[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trained model on CIFAR-10')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型权重的路径(.npz文件)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='数据路径')
    parser.add_argument('--hidden_size', type=int, required=True,
                       help='必须与训练时使用的hidden_size一致')
    parser.add_argument('--activation', type=str, default='relu',
                       help='必须与训练时使用的激活函数一致')
    
    args = parser.parse_args()

    # 初始化模型（参数必须与训练时一致）
    model = Model(
        input_size=3072,  # CIFAR-10固定输入尺寸
        hidden_size=args.hidden_size,
        output_size=10,
        activation=args.activation
    )
    model.load(args.model_path)

    # 加载测试数据
    test_data, test_labels = load_test_data(args.data_path)

    # 计算准确率
    accuracy = compute_accuracy(model, test_data, test_labels)
    
    print(f'Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'Model Info: hidden_size={args.hidden_size}, activation={args.activation}')