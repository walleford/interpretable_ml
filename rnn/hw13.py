import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout,SpatialDropout1D,Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""
    Class to create a recurrent neural network.
    Take in parameters to set for each layer/how many layers as well
"""
class RecurrentNeuralNetwork():
    def __init__(self, lstm_units, embedding_dim, max_phrase_length, x_train, y_train, x_test, y_test):
        self.units = lstm_units
        self.embedding_dim = embedding_dim
        self.max_phrase = max_phrase_length
        self.max_number_words = 50000
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def build(self):
        self.model = Sequential([
                Embedding(input_length=self.max_phrase, 
                                 input_dim=self.max_number_words, 
                                 output_dim=self.embedding_dim),
                SpatialDropout1D(0.1),
                LSTM(units=self.units, activation='relu', dropout=0.2, recurrent_dropout=0.2),
                Dense(units=128, activation='relu'),
                Dense(units=2, activation='softmax'),
            ])
        self.model.build(input_shape=(None, self.max_phrase))
        print(self.model.summary())

    def compile(self):
        self.model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
                )

    def train(self, epochs, batch_size):
        self.history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.x_test, self.y_test))

    def evaluate(self):
        self.test_loss, self.test_acc = self.model.evaluate(self.x_test, self.y_test, verbose=2)

    def graph(self):
        epochs = range(1, len(self.history.history['accuracy']) + 1)
        plt.figure(figsize=(10,10))
        plt.plot(epochs, self.history.history['accuracy'], label='accuracy')
        plt.plot(epochs, self.history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.3,1])
        plt.legend(loc='lower right')
        plt.show()

def read_display_data():
    df = pd.read_csv('~/OneDrive/code/ait636/data/Movie Reviews.csv', index_col=0)
    sns.countplot(data=df, x='label')

    plt.xlabel('Label')
    plt.title('Size of classes')
    plt.show()
    return df

def tokenize(max_number_words, df):
    return tokenizer

if __name__ == "__main__":
    lstm_units = [64,128,256]
    dimensions = [75,100]
    phrase_lengths = [50,75,100]

    df = read_display_data()
    print(df.head())
    print(df['text'].isnull().sum())

    for phrase_length in phrase_lengths:
        for lstm_unit in lstm_units:
            for dimension in dimensions:
                print(f"Phrase Length: {phrase_length}")
                print(f"Units        : {lstm_unit}")
                print(f"Dimensions   : {dimension}")
                print("-----------------------------------")

                tokenizer = Tokenizer(num_words=50000,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~',
                          lower=True)
                tokenizer.fit_on_texts(df['text'].values)
                x = tokenizer.texts_to_sequences(df['text'].values)
                x = pad_sequences(x, maxlen=phrase_length)
                y = pd.get_dummies(df['label']).values
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
                print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
                print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
                network = RecurrentNeuralNetwork(lstm_unit, dimension, phrase_length,
                                                 x_train, y_train, x_test, y_test)
                network.build()
                network.compile()
                network.train(15, 64)
                network.evaluate()
                network.graph()


