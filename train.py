import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from model import *
from tqdm import tqdm

def load_cifar10(data_path):
    # 加载训练数据
    train_data = []
    train_labels = []
    for i in range(1, 6):
        file_path = os.path.join(data_path, f'data_batch_{i}')
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            train_data.append(batch['data'])
            train_labels.append(batch['labels'])
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    # 加载测试数据
    test_path = os.path.join(data_path, 'test_batch')
    with open(test_path, 'rb') as f:
        test_batch = pickle.load(f, encoding='latin1')
        test_data = test_batch['data']
        test_labels = np.array(test_batch['labels'])

    # Normalization
    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0

    return train_data, train_labels, test_data, test_labels

def compute_metrics(model, X, y, batch_size=128):
    num_samples = X.shape[0]
    total_loss = 0.0
    correct = 0
    
    # 添加进度条
    with tqdm(total=num_samples, desc="Evaluating", leave=False) as pbar:
        for i in range(0, num_samples, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            y_pred = model.forward(X_batch)
            loss = model.cross_entropy_loss(y_pred, y_batch)
            total_loss += loss * X_batch.shape[0]
            preds = np.argmax(y_pred, axis=1)
            correct += np.sum(preds == y_batch)
            pbar.update(X_batch.shape[0])
    
    avg_loss = total_loss / num_samples
    accuracy = correct / num_samples
    return avg_loss, accuracy

def plot_metrics(train_losses, test_losses, train_accs, test_accs, save_path):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='基于CIFAR-10数据集的神经网络训练')
    parser.add_argument('--epochs', type=int, required=True, help='训练的轮数')
    parser.add_argument('--data_path', type=str, required=True, help='CIFAR-10数据路径')
    parser.add_argument('--model_save_path', type=str, required=True, help='Model存储路径')
    parser.add_argument('--plot_save_path', type=str, required=True, help='存储loss和accuracy图的路径')
    args = parser.parse_args()

    # 加载数据
    train_data, train_labels, test_data, test_labels = load_cifar10(args.data_path)

    # 模型参数
    input_size = 3072
    hidden_size = 512
    output_size = 10
    activation = 'relu'
    l2_lambda = 0.0001
    lr = 0.01
    batch_size = 128
    lr_decay = 0.95

    # 模型初始化
    model = Model(input_size, hidden_size, output_size, activation=activation, l2_lambda=l2_lambda, lr=lr)

    # 训练
    best_test_acc = 0.0  
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(args.epochs):
        print(f'\nEpoch [{epoch+1}/{args.epochs}]')

        permutation = np.random.permutation(len(train_data))
        train_data_shuffled = train_data[permutation]
        train_labels_shuffled = train_labels[permutation]

        # 批次训练添加进度条
        batch_iter = range(0, len(train_data), batch_size)
        with tqdm(batch_iter, desc=f"Epoch {epoch+1}", leave=False) as batch_pbar:
            for i in batch_pbar:
                X_batch = train_data_shuffled[i:i+batch_size]
                y_batch = train_labels_shuffled[i:i+batch_size]

                model.forward(X_batch)
                model.backward(X_batch, y_batch)
                model.update_params()

                # 实时更新批次进度条信息
                batch_pbar.set_postfix({
                    "Batch Loss": model.cross_entropy_loss(model.a2, y_batch)
                })

        # 计算指标
        train_loss, train_acc = compute_metrics(model, train_data, train_labels)
        test_loss, test_acc = compute_metrics(model, test_data, test_labels)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # 学习率衰减
        lr *= lr_decay

        if test_acc > best_test_acc:
            model.save(args.model_save_path)
            print(  # 使用write避免干扰进度条
                f"已保存更好的模型! Test Acc: {test_acc:.4f} (之前最优的Test Acc: {best_test_acc:.4f})"
            )
            best_test_acc = test_acc

        # 打印
        print(  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    # 保存模型和训练情况
    plot_metrics(train_losses, test_losses, train_accs, test_accs, args.plot_save_path)

if __name__ == '__main__':
    main()