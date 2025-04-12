import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

class ModelVisualizer:
    def __init__(self, model_path, hidden_size, activation, output_dir):
        self.model_path = model_path
        self.hidden_size = hidden_size
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型参数
        self.model = self.load_model()
    
    def load_model(self):
        """加载模型参数（复用训练代码中的模型结构）"""
        class TempModel:
            def __init__(self):
                self.W1 = None
                self.b1 = None
                self.W2 = None
                self.b2 = None
            def load(self, path):
                data = np.load(path)
                self.W1 = data['W1']
                self.b1 = data['b1']
                self.W2 = data['W2']
                self.b2 = data['b2']
        
        model = TempModel()
        model.load(self.model_path)
        return model

    def plot_weight_distributions(self):
        """绘制权重分布直方图"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.model.W1.flatten(), bins=50, alpha=0.7)
        plt.title('W1 Weight Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(self.model.W2.flatten(), bins=50, alpha=0.7, color='orange')
        plt.title('W2 Weight Distribution')
        plt.xlabel('Weight Value')
        
        plt.savefig(os.path.join(self.output_dir, 'weight_distributions.png'))
        plt.close()

    def visualize_hidden_units(self, num_units=20):
        """可视化隐藏层神经元权重模式"""
        # 将权重转换为图像格式 (32, 32, 3)
        weights = self.model.W1.T  # 转置后形状为 [hidden_size, 3072]
        
        plt.figure(figsize=(15, 8))
        plt.suptitle(f'First Layer Weight Patterns (First {num_units} Units)', y=1.02)
        
        for i in range(num_units):
            plt.subplot(4, 5, i+1)
            pixel_data = weights[i].reshape(32, 32, 3)
            
            # 归一化到 [0,1] 范围
            pixel_data = (pixel_data - pixel_data.min()) / (pixel_data.max() - pixel_data.min())
            plt.imshow(pixel_data)
            plt.axis('off')
            plt.title(f'Unit {i+1}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'hidden_unit_patterns.png'), bbox_inches='tight')
        plt.close()

    def plot_weight_matrix(self, layer=1, max_units=64):
        """绘制权重矩阵热力图"""
        weights = self.model.W1 if layer == 1 else self.model.W2
        
        # 限制显示范围
        display_weights = weights[:max_units, :max_units] if layer == 1 else weights[:, :max_units]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(display_weights, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.title(f'Weight Matrix (Layer {layer})')
        plt.xlabel('Input Units' if layer == 1 else 'Hidden Units')
        plt.ylabel('Output Units' if layer == 1 else 'Class Units')
        
        plt.savefig(os.path.join(self.output_dir, f'weight_matrix_layer{layer}.png'))
        plt.close()

    def plot_bias_distribution(self):
        """绘制偏置项分布图"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.model.b1.flatten(), bins=30, alpha=0.7)
        plt.title('Hidden Layer Bias Distribution')
        plt.xlabel('Bias Value')
        
        plt.subplot(1, 2, 2)
        plt.hist(self.model.b2.flatten(), bins=30, alpha=0.7, color='orange')
        plt.title('Output Layer Bias Distribution')
        plt.xlabel('Bias Value')
        
        plt.savefig(os.path.join(self.output_dir, 'bias_distributions.png'))
        plt.close()

    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        self.plot_weight_distributions()
        self.visualize_hidden_units()
        self.plot_weight_matrix(layer=1)
        self.plot_weight_matrix(layer=2)
        self.plot_bias_distribution()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Neural Network Parameters')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model .npz file')
    parser.add_argument('--hidden_size', type=int, required=True,
                       help='必须与训练时使用的hidden_size一致')
    parser.add_argument('--activation', type=str, default='relu',
                       help='必须与训练时使用的激活函数一致')
    parser.add_argument('--output_dir', type=str, default='model_visualization',
                       help='Output directory for visualization images')
    
    args = parser.parse_args()
    
    visualizer = ModelVisualizer(
        args.model_path,
        args.hidden_size,
        args.activation,
        args.output_dir
    )
    visualizer.generate_all_visualizations()
    print(f"Visualizations saved to {args.output_dir} directory")