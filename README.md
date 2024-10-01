# 基于 Seq2Seq 的中英文翻译模型

本项目使用 PyTorch 实现了一个基于 Seq2Seq（编码器-解码器架构）的中英文翻译模型。通过训练该模型，您可以输入英文句子，模型将输出对应的中文翻译。

## 项目目录

```
├── data
│   ├── cmn.txt                # 中英文句子对数据集
│   ├── english_sentences.txt  # 提取的英文句子
│   └── chinese_sentences.txt  # 提取的中文句子
├── main.py                    # 主程序代码
└── README.md                  # 项目说明文件
```

## 环境要求

- Python 3.x
- PyTorch
- TorchText
- Pandas
- Scikit-learn

## 安装依赖

在开始之前，请确保安装了以下依赖库：

```bash
pip install torch torchtext pandas scikit-learn
```

## 数据集

请将 `cmn.txt` 数据集放置在 `data/` 目录下。该文件包含了中英文句子对，每行一个句子对，英文和中文句子以制表符分隔。

您可以从以下链接下载数据集：[datasets](http://www.manythings.org/anki/)

## 运行步骤

### 1. 数据预处理

- **目的**：从原始数据集中提取英文和中文句子，并保存为单独的文件。
- **操作**：运行代码中的数据预处理部分，生成 `english_sentences.txt` 和 `chinese_sentences.txt`。

### 2. 数据加载与分词

- **目的**：加载预处理后的数据，对句子进行分词，并构建英文和中文的词汇表。
- **操作**：代码将自动执行分词和词汇表构建。

### 3. 模型构建

- **目的**：定义编码器、解码器和 Seq2Seq 模型。
- **操作**：代码中已定义了模型的结构，包括参数初始化。

### 4. 模型训练与验证

- **目的**：训练模型并使用验证集评估性能。
- **操作**：运行训练循环，默认训练 10 个 epoch。训练过程中将输出每个 epoch 的训练损失和验证损失。

### 5. 测试与推理

- **目的**：使用训练好的模型进行翻译测试。
- **操作**：在训练完成后，您可以输入英文句子，模型将输出对应的中文翻译。

## 使用方法

1. **确保数据集存在**

    请确保 `data/cmn.txt` 文件存在。如未存在，请下载并放置在 `data/` 目录下。

2. **运行主程序**

    ```bash
    python main.py
    ```

3. **输入英文句子进行测试**

    在程序运行结束后，您可以按照提示输入英文句子，模型将返回中文翻译。

    ```
    请输入英文句子（输入 'quit' 退出）：How are you?
    中文翻译: 你好吗？
    ```

## 模型说明

- **编码器（Encoder）**：使用双层 LSTM，将输入的英文句子编码为上下文向量。
- **解码器（Decoder）**：使用双层 LSTM，根据编码器的输出逐步生成中文句子。
- **Seq2Seq 模型**：整合编码器和解码器，实现序列到序列的翻译。

## 参数设置

- **输入词汇表大小（INPUT_DIM）**：根据英文词汇表的大小自动确定。
- **输出词汇表大小（OUTPUT_DIM）**：根据中文词汇表的大小自动确定。
- **词嵌入维度（ENC_EMB_DIM、DEC_EMB_DIM）**：默认设置为 256。
- **隐藏层维度（HID_DIM）**：默认设置为 512。
- **LSTM 层数（N_LAYERS）**：默认设置为 2。
- **Dropout 概率（ENC_DROPOUT、DEC_DROPOUT）**：默认设置为 0.5。

您可以根据需要调整这些参数以改进模型性能。

## 注意事项

- **训练时间**：由于数据集和模型的复杂性，训练过程可能需要较长时间。建议在 GPU 环境下运行以加快训练速度。
- **模型效果**：由于模型的简单性和数据集的限制，翻译结果可能不够理想。您可以通过增加训练数据、调整模型参数或引入注意力机制等方法改进模型性能。
- **改进建议**：
    - 使用更大的数据集进行训练。
    - 引入注意力机制（Attention）。
    - 使用预训练的词向量。
    - 增加模型的层数或隐藏层维度。

## 参考资料

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [TorchText 文档](https://pytorch.org/text/stable/index.html)
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

## 许可证

本项目仅供学习和研究使用。请勿将本项目用于任何商业用途。

## 联系方式

如果您在使用过程中有任何问题，欢迎提交 Issue 或联系项目维护者。
