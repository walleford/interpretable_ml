from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

class Conv_NN():
    def __init__(self, layers, img_size, batch_size):
        self.layers = layers
        self.img_size = img_size
        self.batch_size = batch_size

    def build_base(self):
       input_shape = (self.img_size, self.img_size, 3)
       if self.layers == 2:
          self.model = Sequential([
              Input(shape=input_shape),
              Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
              MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
              Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
              MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
              Flatten(),
              Dense(units=64, activation='relu'),
              Dense(units=2, activation='softmax')

          ])
       elif self.layers == 3:
          self.model = Sequential([
              Input(shape=input_shape),
              Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
              MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
              Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
              MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
              Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
              MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
              Flatten(),
              Dense(units=64, activation='relu'),
              Dense(units=2, activation='softmax')
          ])
       elif self.layers == 1:
          self.model = Sequential([
              Input(shape=input_shape),
              Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
              MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
              Flatten(),
              Dense(units=64, activation='relu'),
              Dense(units=2, activation='softmax')
          ])

       self.model.summary()

    def compile(self):
        self.model.compile(optimizer='adam', 
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, train_data, epochs, test_data):
        history = self.model.fit(train_data,
                                 epochs=epochs,
                                 validation_data=test_data,
                                 verbose=1)
        self.test_loss, self.test_acc = self.model.evaluate(test_data,steps=len(test_data), verbose=2)
        print(f"Test Accuracy is:    {self.test_acc}")
        return history
        

    def plot(self, history):
        epochs = range(1, len(history.history['accuracy']) + 1)
        plt.figure(figsize=(10,10))
        plt.plot(epochs, history.history['accuracy'], label='accuracy')
        plt.plot(epochs, history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.3,1])
        plt.legend(loc='lower right')
        plt.show()


if __name__ == "__main__":
    # read in the data set here
    # do some cleaning if needed

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    image_size = 112 
    train_data = train_datagen.flow_from_directory(
            '/Users/wallefor/Documents/Cat-Dog/train/',
            target_size=(image_size, image_size),
            batch_size=32,
            class_mode='categorical'
    )

    test_data = validation_datagen.flow_from_directory(
            '/Users/wallefor/Documents/Cat-Dog/test/',
            target_size=(image_size, image_size),
            batch_size=32,
            class_mode='categorical'
    )

    layers = [1,2,3]
    for layer in layers:
        cnn = Conv_NN(layer, image_size, 160)
        cnn.build_base()
        cnn.compile()
        history = cnn.train(train_data, 15, test_data)
        cnn.plot(history)

