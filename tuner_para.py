import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sentencepiece as spm
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFAlbertModel
from tensorflow.keras.callbacks import ModelCheckpoint

# Initialize tokenizer
tokenizer = spm.SentencePieceProcessor(model_file='./custom_tokenizer/custom_tokenizer.model')


# Data preparation
def load_and_prepare_data():
    sequences = []
    labels = []

    with open('./output.txt', 'r') as file:
        next(file)
        for line in file:
            sequence, label = line.strip().split('\t')
            sequences.append(tokenizer.encode(sequence, out_type=int))
            labels.append(int(label))

    max_sequence_length = 512
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', value=0)

    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        padded_sequences, labels, test_size=0.2, random_state=42)

    return (np.array(train_sequences), np.array(train_labels)), (np.array(val_sequences), np.array(val_labels))


# Model definition with hyperparameter optimization
def build_model(hp):
    # 输入
    input_sequence = layers.Input(shape=(512,), dtype='int32')

    # Albert层
    albert_model = TFAlbertModel.from_pretrained('./albert-custom', from_pt=True)
    albert_output = albert_model(input_sequence)[0]

    # CNN-LSTM结构
    # 一维卷积层
    x = layers.Conv1D(filters=hp.Int('cnn_filters', min_value=32, max_value=256, step=32),
                      kernel_size=hp.Int('cnn_kernel_size', min_value=3, max_value=7, step=2),
                      activation='relu')(albert_output)

    # 最大池化层
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Dropout层
    x = layers.Dropout(hp.Float('cnn_dropout', min_value=0.0, max_value=0.7, step=0.1))(x)

    # 双向LSTM层
    x = layers.Bidirectional(layers.LSTM(units=hp.Int('lstm_units', min_value=32, max_value=256, step=32),
                                         return_sequences=True))(x)

    # 获取序列的最后一部分作为表示
    sequence_representation = x[:, -1, :]

    # 密集层
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        sequence_representation = layers.Dense(
            units=hp.Int(f'dense_units_{i}', min_value=32, max_value=512, step=32),
            activation='relu'
        )(sequence_representation)

    # 输出层
    output = layers.Dense(1, activation='sigmoid')(sequence_representation)

    # 构建和编译模型
    model = keras.Model(inputs=input_sequence, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Load the data
(train_data, train_labels), (val_data, val_labels) = load_and_prepare_data()

# Initialize the tuner for hyperparameter optimization
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='hyperparam_opt',
    project_name='albert_opt'
)

# Start the hyperparameter search
tuner.search(
    train_data, train_labels,
    validation_data=(val_data, val_labels),
    epochs=5
)

# Print the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best number of dense layers: {best_hps.get('num_dense_layers')}")
for i in range(best_hps.get('num_dense_layers')):
    print(f"Best number of units for dense layer {i}: {best_hps.get(f'dense_units_{i}')}")
print(f"Best learning rate: {best_hps.get('learning_rate')}")
