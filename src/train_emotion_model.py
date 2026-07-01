import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import os

def build_emotion_cnn(input_shape=(48, 48, 1), num_classes=7):
    """
    Builds a Convolutional Neural Network for FER2013 Emotion Detection.
    """
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("[INFO] Building CNN Architecture...")
    model = build_emotion_cnn()
    model.summary()
    
    # Placeholder for training logic (Requires FER2013 dataset to be downloaded locally)
    print("[INFO] Ready for training on FER2013 dataset.")
    # model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
    # model.save('../models/emotion_model.h5')