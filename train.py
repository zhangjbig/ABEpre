import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import AUC
from transformers import TFAlbertModel
import matplotlib.pyplot as plt
import sentencepiece as spm
import tensorflow as tf
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Define your tokenizer here
tf.random.set_seed(1469)
tokenizer = spm.SentencePieceProcessor(model_file='./custom_tokenizer/custom_tokenizer.model')

# Define the amino acid to integer mapping
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

# Define padding function for one-hot encoded sequences
def pad_one_hot_sequences(one_hot_sequences, max_sequence_length):
    padded_sequences = np.zeros((len(one_hot_sequences), max_sequence_length, len(amino_acid_to_int)))
    for idx, seq in enumerate(one_hot_sequences):
        sequence_length = min(len(seq), max_sequence_length)
        padded_sequences[idx, :sequence_length, :] = seq[:sequence_length]
    return padded_sequences

# Function to load and prepare data
def load_and_prepare_data(test_size=0.2, max_sequence_length=512):
    sequences = []
    labels = []

    with open('./train_data/train.txt', 'r') as file:
        next(file)
        for line in file:
            sequence, label = line.strip().split('\t')
            sequences.append(sequence)
            labels.append(int(label))

        # 分词器编码和独热编码
    tokenized_sequences = [tokenizer.encode(seq, out_type= int) for seq in sequences]
    one_hot_sequences = [one_hot_encode(seq) for seq in sequences]

    # 填充序列以获得统一长度
    tokenized_sequences = pad_sequences(tokenized_sequences, maxlen=max_sequence_length, padding='post')
    one_hot_sequences = pad_sequences(one_hot_sequences, maxlen=max_sequence_length, padding='post')
    print(1)
    # 划分数据为训练集和验证集
    train_tokenized_sequences, val_tokenized_sequences, train_one_hot_sequences, val_one_hot_sequences, train_labels, val_labels = train_test_split(
        tokenized_sequences, one_hot_sequences, labels, test_size=test_size, random_state=42)

    # 过采样以平衡类别
    ros = RandomOverSampler(random_state=42)
    train_indices_resampled, train_labels_resampled = ros.fit_resample(np.arange(len(train_labels)).reshape(-1, 1),
                                                                       train_labels)
    train_indices_resampled = train_indices_resampled.flatten()
    print(2)
    # 应用过采样的索引
    train_tokenized_sequences = train_tokenized_sequences[train_indices_resampled]
    train_one_hot_sequences = train_one_hot_sequences[train_indices_resampled]
    train_labels = np.array(train_labels)[train_indices_resampled]
    val_labels = np.array(val_labels)
    # 洗牌数据
    train_tokenized_sequences, train_one_hot_sequences, train_labels = shuffle(
        train_tokenized_sequences, train_one_hot_sequences, train_labels)
    print("Train tokenized sequences type:", type(train_tokenized_sequences))
    print("Train tokenized sequences shape:", train_tokenized_sequences.shape)

    print("Train one-hot sequences type:", type(train_one_hot_sequences))
    print("Train one-hot sequences shape:", train_one_hot_sequences.shape)

    print("Validation tokenized sequences type:", type(val_tokenized_sequences))
    print("Validation tokenized sequences shape:", val_tokenized_sequences.shape)

    print("Validation one-hot sequences type:", type(val_one_hot_sequences))
    print("Validation one-hot sequences shape:", val_one_hot_sequences.shape)

    print("Train labels type:", type(train_labels))
    print("Train labels shape:", train_labels.shape)

    print("Validation labels type:", type(val_labels))
    print("Validation labels shape:", val_labels.shape)

    return (train_tokenized_sequences, train_one_hot_sequences, train_labels), (
    val_tokenized_sequences, val_one_hot_sequences, val_labels)


# Function to build and train the model
def build_and_train_model(test_size, iteration, max_sequence_length=512):
    (train_tokenized_sequences, train_one_hot_sequences, train_labels), (val_tokenized_sequences, val_one_hot_sequences, val_labels) = load_and_prepare_data(test_size, max_sequence_length)

    # Define the model with two branches
    # Branch 1: Tokenized sequences
    tokenized_input = layers.Input(shape=(max_sequence_length,), dtype='int32', name='tokenized_input')
    albert_model = TFAlbertModel.from_pretrained('./albert-custom', from_pt=True)
    tokenized_embedding = albert_model(tokenized_input)[0]
    x1 = layers.Conv1D(64, 3, activation='relu')(tokenized_embedding)
    x1 = layers.MaxPooling1D(2)(x1)
    x1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x1)
    x1 = layers.Bidirectional(layers.LSTM(32))(x1)

    # Branch 2: One-hot encoded sequences
    one_hot_input = layers.Input(shape=(max_sequence_length, len(amino_acid_to_int)), dtype='float32', name='one_hot_input')
    x2 = layers.Conv1D(64, 3, activation='relu')(one_hot_input)
    x2 = layers.MaxPooling1D(2)(x2)
    x2 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x2)
    x2 = layers.Bidirectional(layers.LSTM(32))(x2)

    # Merge branches
    merged = layers.concatenate([x1, x2])
    z = layers.Dense(480, activation='relu')(merged)
    z = layers.Dropout(0.25)(z)
    z = layers.Dense(32, activation='relu')(z)
    z = layers.Dense(32, activation='relu')(z)
    output = layers.Dense(1, activation='sigmoid')(z)

    # Create the model
    model = keras.Model(inputs=[tokenized_input, one_hot_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC(name='auc'), 'accuracy'])

    # Train the model
    history = model.fit(
        [train_tokenized_sequences, train_one_hot_sequences], train_labels,
        validation_data=([val_tokenized_sequences, val_one_hot_sequences], val_labels),
        epochs=5,
        batch_size=64
    )

    # Save the model and plot the training history
    model_save_path = f'./model/model_{test_size}/model_test_size_{test_size}_iter_{iteration}'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model.save(os.path.join(model_save_path, 'model.h5'))
    hisdf = pd.DataFrame(history.history)
    hisdf.to_csv(os.path.join(model_save_path, 'history.csv'),index=False)
    # Plot the AUC and loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Training and Validation AUC')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'training_history.png'))

# Run the training for different test sizes and iterations
for test_size in [0.1,0.2,0.4,0.5]:
    for iteration in range(0,10):
        build_and_train_model(test_size, iteration)
