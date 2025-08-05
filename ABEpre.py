import argparse
import numpy as np
import re
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from transformers import TFAlbertModel
import sentencepiece as spm
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 假设模型路径和分词器路径是固定的
model_path = './model/model_0.1/model_test_size_0.1_iter_8/model.h5'
tokenizer_model = './custom_tokenizer/custom_tokenizer.model'

# 加载模型和分词器
with custom_object_scope({'TFAlbertModel':TFAlbertModel}):
    model = load_model(model_path)
tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model)

amino_acid_to_int = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3,
    'F': 4, 'G': 5, 'H': 6, 'I': 7,
    'K': 8, 'L': 9, 'M': 10, 'N': 11,
    'P': 12, 'Q': 13, 'R': 14, 'S': 15,
    'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

# Define one-hot encoding function
def one_hot_encode(sequence):
    encoding = np.zeros((len(sequence), len(amino_acid_to_int)))
    for idx, char in enumerate(sequence):
        if char in amino_acid_to_int:
            encoding[idx, amino_acid_to_int[char]] = 1
    return encoding

def pad_one_hot_sequences(one_hot_sequences, max_sequence_length):
    padded_sequences = np.zeros((len(one_hot_sequences), max_sequence_length, len(amino_acid_to_int)))
    for idx, seq in enumerate(one_hot_sequences):
        sequence_length = min(len(seq), max_sequence_length)
        padded_sequences[idx, :sequence_length, :] = seq[:sequence_length]
    return padded_sequences

# 读取并处理输入文件
def read_and_process_input(infile, seq_type):
    with open(infile, 'r') as file:
        raw_data = file.read()

    sequences = re.findall(r">(.*?)\n([^>]*)", raw_data, re.DOTALL)
    if not sequences:  # 如果没有找到以>开头的序列标签，就用数字标记
        sequences = [('Seq_{}'.format(i), seq) for i, seq in enumerate(raw_data.split(), start=1)]

    processed_data = []
    for id, seq in sequences:
        seq = seq.replace('\n', '').replace('\r', '').upper()# 移除换行符
        if seq_type == 'protein' and len(seq) > 8:
            # 切割成长度为8到30的序列
            for start in range(0, len(seq) - 8 + 1):
                for end in range(start+8,min(start + 31, len(seq)+1)):
                #end = min(start + 30, len(seq))
                    sub_seq = seq[start:end]
                    processed_data.append((id, sub_seq))
        elif seq_type == 'peptide' and 8 <= len(seq) <= 30:
            processed_data.append((id, seq))

    return processed_data

# 主函数
def main(args):
    seq_type = args.type.lower()
    infile = args.infile
    outfile = args.outfile

    # 读取输入文件并处理
    processed_data = read_and_process_input(infile, seq_type)
    # 预测并写入输出文件
    with open(outfile, 'w') as f_out:
        for (id, peptide) in processed_data:
            print(peptide)
            tokenized_seq = tokenizer.encode(peptide, out_type=int)
            print(tokenized_seq)
            one_hot_seq = one_hot_encode(peptide)
            print(one_hot_seq)
            tokenized_padded = pad_sequences([tokenized_seq], maxlen=512, padding='post', truncating='post')
            one_hot_padded = pad_one_hot_sequences([one_hot_seq], 512)
            score = model.predict_on_batch([tokenized_padded, one_hot_padded])[0][0]
            f_out.write(f"{id}\t{peptide}\t{score}\n")

# 解析命令行参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequence Prediction Tool")
    parser.add_argument("-type", choices=['protein', 'peptide'], required=True, help="Type of the sequences to process")
    parser.add_argument("-infile", required=True, help="Input file path")
    parser.add_argument("-outfile", required=True, help="Output file path")
    args = parser.parse_args()
    main(args)
