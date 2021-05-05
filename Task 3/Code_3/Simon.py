# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#--------------------------------------------------------------------------------------------------
# Description Simon.py
# Code by sitobler for Task3
#
#--------------------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def bdpsimon_train2letters(pandas_traindata):
    seq_as_word=pandas_traindata['Sequence'].to_numpy()
    seq=[]
    for entry in seq_as_word:
        seq.append(list(entry))

    seq=np.asarray(seq)
    return[seq, seq_as_word, pandas_traindata['Active'].to_numpy()]

def keras_getmodel():
    inputs = keras.Input(shape=(4, ))
    dense = layers.Dense(10, activation='relu')
    x = dense(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    model.summary()

    x_train = x_train[5000:, :]
    y_train = y_train[5000:]
    x_test = x_train[0:4999, :]
    y_test = y_train[0:4999]

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )

    history = model.fit(x_train, y_train, epochs=2, validation_split=0.2)

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    print("Histors type:", type(history))

    return model


def keras_getmodel_fullseq(vec_dim, x_train, y_train):
    num_dist_words = 194481  # 21**4

    inputs = keras.Input(shape=(1, ))
    embed_inputs = layers.Embedding(num_dist_words, vec_dim)(inputs)
    dense = layers.Dense(10, activation='relu')
    x = dense(embed_inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    model.summary()

    x_train = x_train[5000:]
    y_train = y_train[5000:]
    x_test = x_train[0:4999]
    y_test = y_train[0:4999]

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )

    history = model.fit(x_train, y_train, epochs=2, validation_split=0.2)

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    print("Histors type:", type(history))

    return model

def keras_test(vec_dim, x_train, y_train):
    num_dist_words = 194481  # 21**4

    model=keras.Sequential()
    model.add(layers.Embedding(num_dist_words, vec_dim))

    model.summary()

    x_train = x_train[5000:]
    y_train = y_train[5000:]
    x_test = x_train[0:4999]
    y_test = y_train[0:4999]

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )

    output=model.predict(x_train)
    print(output)

    return model
