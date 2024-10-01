# 基于 Seq2Seq 的中英文翻译模型

# 导入必要的库
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Step 1: 数据预处理
# 读取原始数据并提取英文和中文句子

file_path = 'data/cmn.txt'  # 请确保数据文件位于该路径下

# 读取文件并处理每一行，提取英文和中文句子
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 每行数据使用制表符分割，提取英文和中文部分
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            english_sentence = parts[0].strip()
            chinese_sentence = parts[1].strip()
            data.append([english_sentence, chinese_sentence])

# 创建 DataFrame 保存提取的句子
df = pd.DataFrame(data, columns=['English', 'Chinese'])

# 将处理后的英文和中文句子分别保存为两个文件
df['English'].to_csv('data/english_sentences.txt', index=False, header=False)
df['Chinese'].to_csv('data/chinese_sentences.txt', index=False, header=False)

# 显示前五行数据
print(df.head())

# Step 2: 数据加载与分词

# 定义英文和中文的分词器
tokenizer_en = get_tokenizer('basic_english')

# 中文分词器：将每个汉字作为一个 token
def tokenizer_zh(text):
    return list(text)

# 构建词汇表的函数
def build_vocab(sentences, tokenizer):
    """
    根据给定的句子列表和分词器构建词汇表。
    :param sentences: 句子列表
    :param tokenizer: 分词器函数
    :return: 词汇表对象
    """
    def yield_tokens(sentences):
        for sentence in sentences:
            yield tokenizer(sentence)
    vocab = build_vocab_from_iterator(yield_tokens(sentences), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    vocab.set_default_index(vocab['<unk>'])  # 设置默认索引为 <unk>
    return vocab

# 从文件中加载句子
with open('data/english_sentences.txt', 'r', encoding='utf-8') as f:
    english_sentences = [line.strip() for line in f]

with open('data/chinese_sentences.txt', 'r', encoding='utf-8') as f:
    chinese_sentences = [line.strip() for line in f]

# 构建英文和中文的词汇表
en_vocab = build_vocab(english_sentences, tokenizer_en)
zh_vocab = build_vocab(chinese_sentences, tokenizer_zh)

print(f'英文词汇表大小：{len(en_vocab)}')
print(f'中文词汇表大小：{len(zh_vocab)}')

# 定义将句子转换为索引序列的函数
def process_sentence(sentence, tokenizer, vocab):
    """
    将句子转换为索引序列，并添加 <bos> 和 <eos>
    :param sentence: 输入句子
    :param tokenizer: 分词器函数
    :param vocab: 对应的词汇表
    :return: 索引序列
    """
    tokens = tokenizer(sentence)
    tokens = ['<bos>'] + tokens + ['<eos>']
    indices = [vocab[token] for token in tokens]
    return indices

# 将所有句子转换为索引序列
en_sequences = [process_sentence(sentence, tokenizer_en, en_vocab) for sentence in english_sentences]
zh_sequences = [process_sentence(sentence, tokenizer_zh, zh_vocab) for sentence in chinese_sentences]

# 查看示例句子的索引序列
print("示例英文句子索引序列：", en_sequences[0])
print("示例中文句子索引序列：", zh_sequences[0])

# 创建数据集和数据加载器

class TranslationDataset(Dataset):
    def __init__(self, src_sequences, trg_sequences):
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.src_sequences[idx]), torch.tensor(self.trg_sequences[idx])

def collate_fn(batch):
    """
    自定义的 collate_fn，用于将批次中的样本进行填充对齐
    """
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(src_sample)
        trg_batch.append(trg_sample)
    src_batch = pad_sequence(src_batch, padding_value=en_vocab['<pad>'])
    trg_batch = pad_sequence(trg_batch, padding_value=zh_vocab['<pad>'])
    return src_batch, trg_batch

# 创建数据集对象
dataset = TranslationDataset(en_sequences, zh_sequences)

# 划分训练集和验证集
train_data, val_data = train_test_split(dataset, test_size=0.1)

# 创建数据加载器
batch_size = 32
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Step 3: Seq2Seq 模型构建

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        """
        初始化编码器
        :param input_dim: 输入词汇表的大小（英文词汇表大小）
        :param emb_dim: 词嵌入维度
        :param hid_dim: 隐藏层维度
        :param n_layers: LSTM 层数
        :param dropout: Dropout 概率
        """
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        前向传播
        :param src: [src_len, batch_size]
        :return: hidden, cell
        """
        embedded = self.dropout(self.embedding(src))  # [src_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)  # outputs: [src_len, batch_size, hid_dim]
        return hidden, cell  # hidden/cell: [n_layers, batch_size, hid_dim]

# 定义解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        """
        初始化解码器
        :param output_dim: 输出词汇表大小（中文词汇表大小）
        :param emb_dim: 词嵌入维度
        :param hid_dim: 隐藏层维度
        :param n_layers: LSTM 层数
        :param dropout: Dropout 概率
        """
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        """
        前向传播
        :param input: [batch_size]
        :param hidden: [n_layers, batch_size, hid_dim]
        :param cell: [n_layers, batch_size, hid_dim]
        :return: prediction, hidden, cell
        """
        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # [1, batch_size, emb_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))  # output: [1, batch_size, hid_dim]
        prediction = self.fc_out(output.squeeze(0))  # [batch_size, output_dim]
        return prediction, hidden, cell

# 定义 Seq2Seq 模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        """
        初始化 Seq2Seq 模型
        :param encoder: 编码器对象
        :param decoder: 解码器对象
        :param device: 设备（CPU 或 GPU）
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        前向传播
        :param src: [src_len, batch_size]
        :param trg: [trg_len, batch_size]
        :param teacher_forcing_ratio: 教师强制比率
        :return: outputs, [trg_len, batch_size, output_dim]
        """
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # 存储解码器输出
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # 编码器输出
        hidden, cell = self.encoder(src)

        # 解码器初始输入为 <bos> token
        input = trg[0, :]  # [batch_size]

        for t in range(1, trg_len):
            # 解码器前向传播
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            # 决定是否使用教师强制
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)  # [batch_size]
            input = trg[t] if teacher_force else top1

        return outputs

# 初始化模型参数
INPUT_DIM = len(en_vocab)
OUTPUT_DIM = len(zh_vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# 实例化编码器、解码器和 Seq2Seq 模型
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

# 初始化模型参数
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

# 定义损失函数和优化器
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=zh_vocab['<pad>'])

# Step 4: 模型训练与验证

# 定义训练函数
def train(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        # output: [trg_len, batch_size, output_dim]
        # trg: [trg_len, batch_size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)  # 去掉 <bos>
        trg = trg[1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 防止梯度爆炸
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# 定义验证函数
def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)  # 关闭教师强制
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)  # 去掉 <bos>
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# 开始训练模型
n_epochs = 10
clip = 1

for epoch in range(n_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion, clip)
    val_loss = evaluate(model, val_dataloader, criterion)
    print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')

# Step 5: 测试与推理

# 定义翻译函数
def translate_sentence(sentence, model, en_vocab, zh_vocab, tokenizer_en, max_len=50):
    """
    翻译英文句子为中文
    :param sentence: 英文句子（字符串）
    :param model: 训练好的 Seq2Seq 模型
    :param en_vocab: 英文词汇表
    :param zh_vocab: 中文词汇表
    :param tokenizer_en: 英文分词器
    :param max_len: 最大翻译长度
    :return: 中文翻译（字符串）
    """
    model.eval()
    tokens = tokenizer_en(sentence)
    tokens = ['<bos>'] + tokens + ['<eos>']
    src_indices = [en_vocab[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(device)  # [src_len, 1]
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
    trg_indices = [zh_vocab['<bos>']]
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)
        if pred_token == zh_vocab['<eos>']:
            break
    trg_tokens = [zh_vocab.lookup_token(idx) for idx in trg_indices]
    return ''.join(trg_tokens[1:-1])  # 去除 <bos> 和 <eos>

# 示例测试
input_sentence = "How are you?"
translation = translate_sentence(input_sentence, model, en_vocab, zh_vocab, tokenizer_en)
print(f"英文句子: {input_sentence}")
print(f"中文翻译: {translation}")

# 您可以在此处输入其他英文句子进行测试
while True:
    input_sentence = input("请输入英文句子（输入 'quit' 退出）：")
    if input_sentence.lower() == 'quit':
        break
    translation = translate_sentence(input_sentence, model, en_vocab, zh_vocab, tokenizer_en)
    print(f"中文翻译: {translation}")
