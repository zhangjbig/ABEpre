import sentencepiece as spm

# 从你的文件中读取序列
def extract_sequences(file_path):
    with open(file_path, 'r') as file:
        sequences = []
        count = 0
        for line in file:
            if count == 0:
                count = 1
                continue
            else:
                count = 0
                sequences.append(line.strip())
        return sequences

# 保存序列到新的文件中，每个序列占一行
def save_sequences(sequences, output_file):
    with open(output_file, 'w') as file:
        for sequence in sequences:
            file.write(sequence + '\n')

# 你的文件路径
file_path = '../Albert/viral.txt'

# 提取序列
sequences = extract_sequences(file_path)

# 保存序列
output_file = '../Albert/sequences.txt'
save_sequences(sequences, output_file)

# 训练 SentencePiece 分词器
spm.SentencePieceTrainer.train(
    input=output_file,  # 输入文件路径
    model_prefix='custom_tokenizer',  # 输出模型前缀
    vocab_size=30,  # 词汇表大小
    character_coverage=1.0,  # 字符覆盖
    model_type='unigram',  # 模型类型
)

# 现在你可以使用以下命令加载和使用分词器
tokenizer = spm.SentencePieceProcessor(model_file='./custom_tokenizer/custom_tokenizer.model')

# 测试分词器
encoded = tokenizer.encode('ARNDCEQG')
print(encoded)  # 输出: [0, 1, 2, 3, 4, 5, 6, 7]
vocab_size = tokenizer.get_piece_size()

for i in range(vocab_size):
    token=tokenizer.id_to_piece(i)
    token_id=tokenizer.piece_to_id(token)
    print(f"token:{token},token_id:{token_id}")