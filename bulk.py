import os
import re
import multiprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from transformers import TFAlbertModel
import sentencepiece as spm
import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 加载模型和分词器
model_path = './model/model_0.1/model_test_size_0.1_iter_8/model.h5'
tokenizer_model = './custom_tokenizer/custom_tokenizer.model'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
amino_acid_to_int = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3,
    'F': 4, 'G': 5, 'H': 6, 'I': 7,
    'K': 8, 'L': 9, 'M': 10, 'N': 11,
    'P': 12, 'Q': 13, 'R': 14, 'S': 15,
    'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

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

def process_file(file_path):
    with custom_object_scope({'TFAlbertModel':TFAlbertModel}):
        model = load_model(model_path)
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join('../ABCpreout/4/', base_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    for i in range(0, len(lines), 2):
        virus_name = lines[i].strip()
        virus_seq = lines[i + 1].strip()
        all_pre = []
        print(virus_name)
        if os.path.exists(os.path.join(output_dir, f'{virus_name}.txt')):
            print(f"预测结果已存在，跳过 {virus_name}")
            continue
        for start in range(0, len(virus_seq) - 7):
            for end in range(start + 8, min(start + 21, len(virus_seq) + 1)):
                peptide = virus_seq[start:end]
                tokenized_seq = tokenizer.encode(peptide, out_type=int)
                one_hot_seq = one_hot_encode(peptide)
                tokenized_padded = pad_sequences([tokenized_seq], maxlen=512, padding='post', truncating='post')
                one_hot_padded = pad_one_hot_sequences([one_hot_seq], 512)
                score = model.predict_on_batch([tokenized_padded, one_hot_padded])[0][0]
                if score >= 0.5:
                    all_pre.append((peptide,start,score))

        with open(os.path.join(output_dir, f'{virus_name}.txt'), 'w') as f_out:
            f_out.write("peptide\tstart\tscore\n")
            for peptide, start, score in all_pre:
                f_out.write(f"{peptide}\t{start}\t{score}\n")

def main():
    input_dir = '/media/ubuntu/55b53b7a-76ed-4ea2-9e39-eb13d6e00f7b/tcz/4'
    files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if 'xaa' <= file <= 'xbj']
    with multiprocessing.Pool(36) as pool:
        pool.map(process_file, files)

if __name__ == "__main__":
    main()
