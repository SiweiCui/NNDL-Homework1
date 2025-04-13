import argparse
import itertools
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from model import Model
from train import load_cifar10, compute_metrics

def train_single_model(params, train_data, train_labels, test_data, test_labels, epochs=30, batch_size=128):
    """训练单个模型并返回训练指标"""
    model = Model(
        input_size=3072,
        hidden_size=params['hidden_size'],
        output_size=10,
        activation=params['activation'],
        l2_lambda=params['l2_lambda'],
        lr=params['learning_rate']
    )

    for key, value in params.items():
        print(f"{key}: {value}", end="  ")
    
    print("\nTraining...")
    
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    for epoch in range(epochs):

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
        
        # 每个epoch记录指标
        train_loss, train_acc = compute_metrics(model, train_data, train_labels)
        test_loss, test_acc = compute_metrics(model, test_data, test_labels)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    
    return {
        'train_loss': train_losses,
        'test_loss': test_losses,
        'train_acc': train_accs,
        'test_acc': test_accs,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'final_test_acc': test_accs[-1]
    }

def save_results(params, metrics, log_file, output_dir):
    """保存结果到日志文件和生成曲线图"""
    # 保存到日志文件
    log_entry = {
        **params,
        'final_train_loss': metrics['final_train_loss'],
        'final_test_loss': metrics['final_test_loss'],
        'final_test_acc': metrics['final_test_acc'],
        'timestamp': datetime.now().isoformat()
    }
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    # 生成训练曲线图
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['test_loss'], label='Test Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_acc'], label='Train Acc')
    plt.plot(metrics['test_acc'], label='Test Acc')
    plt.title('Accuracy Curves')
    
    # 生成唯一文件名
    param_str = f"lr{params['learning_rate']}_hs{params['hidden_size']}_l2{params['l2_lambda']}_act{params['activation']}"
    plt.savefig(os.path.join(output_dir, f'training_curves_{param_str}.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Grid Search')
    parser.add_argument('--learning_rates', nargs='+', type=float, required=True)
    parser.add_argument('--hidden_sizes', nargs='+', type=int, required=True)
    parser.add_argument('--l2_lambdas', nargs='+', type=float, required=True)
    parser.add_argument('--activation', nargs='+', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='grid_search_results')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'grid_search.log')

    # 加载数据
    train_data, train_labels, test_data, test_labels = load_cifar10(args.data_path)

    # 生成参数网格
    param_grid = {
        'learning_rate': args.learning_rates,
        'hidden_size': args.hidden_sizes,
        'l2_lambda': args.l2_lambdas,
        'activation': args.activation
    }
    keys = param_grid.keys()
    values = (param_grid[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    
    for i in range(len(combinations)):
        
        # 训练模型
        metrics = train_single_model(combinations[i], train_data, train_labels, test_data, test_labels)
        
        # 保存结果
        save_results(combinations[i], metrics, log_file, args.output_dir)

if __name__ == '__main__':
    main()