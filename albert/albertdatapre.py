from Bio import SeqIO
from sklearn.model_selection import train_test_split
import numpy as np

# 定义用于将氨基酸映射到整数的字典
amino_acid_mapping = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
    'X': 20  # X通常用于表示未知或任意氨基酸
}

MAX_SEQUENCE_LENGTH = 512

def is_valid_sequence(seq):
    return all(aa in amino_acid_mapping for aa in seq)

def parse_fasta(file_path):
    with open(file_path, 'r') as file:
        sequences = []
        count = 0
        for line in file:
            line = line.strip()
            if count == 0:
                count = 1
                continue
            else:
                count = 0
                sequences.append(line)
        return sequences

# 解析FASTA文件以提取蛋白质序列和标识符
file_path = './data/viral.txt'
sequences = parse_fasta(file_path)
sequences = [seq for seq in sequences if is_valid_sequence(seq)]

# 将蛋白质序列编码为整数
encoded_sequences = [[amino_acid_mapping[aa] for aa in seq] for seq in sequences]

# 对序列进行填充或截断以确保它们具有相同的长度
padded_sequences = np.zeros((len(encoded_sequences), MAX_SEQUENCE_LENGTH), dtype=int)
for i, seq in enumerate(encoded_sequences):
    if len(seq) > MAX_SEQUENCE_LENGTH:
        padded_sequences[i] = seq[:MAX_SEQUENCE_LENGTH]
    else:
        padded_sequences[i, :len(seq)] = seq

# 分割数据为训练集、验证集和测试集
train_data, test_data = train_test_split(padded_sequences, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# 保存处理后的数据到文件
np.save('train_data.npy', train_data)
np.save('val_data.npy', val_data)
np.save('test_data.npy', test_data)
