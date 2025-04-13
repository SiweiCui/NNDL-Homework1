# 神经网络与深度学习Homework1

## 文件说明

`.py`文件是代码, `.bat`文件是命令行脚本.

`model.py`: 定义了模型.

`train.py`/`train.bat`: 模型训练.

`test.py`/`test.bat`: 导入训练好的模型进行测试.

`grid_search.py`/`grid_search.bat`: 基于网格的参数查找.

`visualization.py`/`visualization.bat`: 导入训练好的模型, 查看参数的分布模式.

## 使用说明

在windows cmd中可直接运行bat脚本. 此处解释如何运行Python文件:

### 1. train.py - 模型训练

**功能**：训练神经网络模型并保存最佳模型权重

**运行命令**：
```bash
python train.py \
  --epochs <训练轮数> \
  --data_path <数据集路径> \
  --model_save_path <模型保存路径> \
  --plot_save_path <训练曲线图保存路径> 
```

**参数说明**：
| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|-------|--------|------|
| `--epochs` | int | 是 | - | 训练的总轮数 |
| `--data_path` | str | 是 | - | CIFAR-10 数据集目录路径 |
| `--model_save_path` | str | 是 | - | 模型权重保存路径 (.npz 文件) |
| `--plot_save_path` | str | 是 | - | 训练曲线图保存路径 (.png 文件) |

### 2. test.py - 模型测试脚本

**功能**：评估训练好的模型在测试集上的性能

**运行命令**：
```bash
python test.py \
  --model_path <模型路径> \
  --data_path <数据集路径> \
  --hidden_size <隐藏层大小> \
  [--activation <激活函数>]
```

**参数说明**：
| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|-------|--------|------|
| `--model_path` | str | 是 | - | 训练好的模型权重路径 (.npz 文件) |
| `--data_path` | str | 是 | - | CIFAR-10 数据集目录路径 |
| `--hidden_size` | int | 是 | - | 隐藏层神经元数量 (必须与训练时一致) |
| `--activation` | str | 否 | 'relu' | 使用的激活函数 (必须与训练时一致) |

### 3. grid-search.py - 超参数网格搜索脚本

**功能**：自动搜索最优超参数组合

**运行命令**：
```bash
python grid-search.py \
  --learning_rates <学习率列表> \
  --hidden_sizes <隐藏层大小列表> \
  --l2_lambdas <L2正则化强度列表> \
  --activation <激活函数列表> \
  --data_path <数据集路径> \
  [--output_dir <输出目录>]
```

**参数说明**：
| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|-------|--------|------|
| `--learning_rates` | float列表 | 是 | - | 要测试的学习率值 (空格分隔) |
| `--hidden_sizes` | int列表 | 是 | - | 要测试的隐藏层大小 (空格分隔) |
| `--l2_lambdas` | float列表 | 是 | - | 要测试的L2正则化强度 (空格分隔) |
| `--activation` | str列表 | 是 | - | 要测试的激活函数 (空格分隔) |
| `--data_path` | str | 是 | - | CIFAR-10 数据集目录路径 |
| `--output_dir` | str | 否 | 'grid_search_results' | 结果保存目录 |

### 4. visualization.py - 模型可视化脚本

**功能**：可视化训练好的模型参数

**运行命令**：
```bash
python visualization.py \
  --model_path <模型路径> \
  --hidden_size <隐藏层大小> \
  [--activation <激活函数>] \
  [--output_dir <输出目录>]
```

**参数说明**：
| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|-------|--------|------|
| `--model_path` | str | 是 | - | 训练好的模型权重路径 (.npz 文件) |
| `--hidden_size` | int | 是 | - | 隐藏层神经元数量 (必须与训练时一致) |
| `--activation` | str | 否 | 'relu' | 使用的激活函数 (必须与训练时一致) |
| `--output_dir` | str | 否 | 'model_visualization' | 可视化结果保存目录 |

## 示例工作流程

1. **训练模型**：
```bash
python train.py --epochs 50 --data_path ./cifar-10-batches-py \
  --model_save_path best_model.npz --plot_save_path training_curves.png
```

2. **测试模型**：
```bash
python test.py --model_path best_model.npz --data_path ./cifar-10-batches-py \
  --hidden_size 512
```

3. **超参数搜索**：
```bash
python grid-search.py --learning_rates 0.1 0.01 0.001 --hidden_sizes 256 512 \
  --l2_lambdas 0.001 0.0001 --activation relu sigmoid --data_path ./cifar-10-batches-py
```

4. **可视化模型**：
```bash
python visualization.py --model_path best_model.npz --hidden_size 512 \
  --output_dir model_viz
```

## 输出说明

1. **训练脚本**：
   - 模型权重文件 (.npz)
   - 训练曲线图 (.png)

2. **测试脚本**

    - 命令行输出模型的效果及模型配置信息.

3. **网格搜索**：
   - 日志文件 (grid_search.log)
   - 每个参数组合的训练曲线图

4. **可视化脚本**：
   - 权重分布图
   - 隐藏单元模式图
   - 权重矩阵热力图
   - 偏置项分布图
