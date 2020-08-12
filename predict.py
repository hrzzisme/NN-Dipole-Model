import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
import glob


def my_RMSE(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = y_pred - y_true
    return K.sqrt(K.mean(K.square(tf.norm(diff, ord='euclidean', axis=-1)), axis=-1))


def mse_and_norm_mse(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = y_pred - y_true
    y_pred_norm = tf.norm(y_pred, ord='euclidean', axis=-1)
    y_true_norm = tf.norm(y_true, ord='euclidean', axis=-1)
    norm_diff = y_pred_norm - y_true_norm
    loss = K.mean(K.square(tf.norm(diff, ord='euclidean', axis=-1)), axis=-1) + K.mean(K.square(norm_diff), axis=-1)
    return loss


def build_model(load_weights=False, load_weights_file=None):
#    model = keras.Sequential([
#        keras.layers.Dense(800, activation=tf.nn.tanh),
#        keras.layers.Dense(800, activation=tf.nn.tanh),
#        keras.layers.Dense(800, activation=tf.nn.tanh),
#        keras.layers.Dense(3)
#    ])
    model = keras.Sequential([
    keras.layers.Dense(300, activation=tf.nn.tanh),
    keras.layers.Dense(200, activation=tf.nn.tanh),
    keras.layers.Dense(100, activation=tf.nn.tanh),
    keras.layers.Dense(30, activation=tf.nn.tanh),
    keras.layers.Dense(3)
    ])
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-3,decay_steps=train_steps*5,
    #                                                         decay_rate=0.99,staircase=True)
    # optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optim = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optim,
                  loss=mse_and_norm_mse,
                  metrics=[my_RMSE])
    if load_weights:
        model.load_weights(load_weights_file)
        print('Load weights from {}'.format(load_weights_file))
    return model


def generate_data(file_name):
    raw_data = pd.read_csv(file_name, header=None, delim_whitespace=True)
    data_x = raw_data.iloc[:, 1:-13].values
    print(data_x.shape)
    data_y = raw_data.iloc[:, -12:-9].values
    rotation_matrix = raw_data.iloc[:, -9:].values
    row_num = len(data_y)
    rotation_matrix = rotation_matrix.reshape(row_num, 3, 3)
    return data_x, data_y, rotation_matrix


files_folder = '' #path of folder which stores test data file
test_data_files = glob.glob(files_folder + "*.dat")
LEARNING_RATE = 1e-4
ckpt = tf.train.latest_checkpoint('./TF_weights/')
model = build_model(load_weights=True, load_weights_file=ckpt)
#print(model.summary())
for file_name in test_data_files:
    prefix = file_name.split('/')[-1].split('.')[0]
    data_x, data_y, rotation_matrix = generate_data(file_name)
    predictions = model.predict(np.array(data_x))
    with open('{}-predict.dat'.format(prefix), 'w') as f:
        for index, y_pred in enumerate(predictions):
            intensity = np.linalg.norm(y_pred)
            y_true = data_y[index]
            R = rotation_matrix[index]
            R_inv = np.linalg.inv(R)
            rotated_y_pred = np.matmul(R_inv, y_pred)
            rotated_y_true = np.matmul(R_inv, y_true)
            true_intensity = np.linalg.norm(rotated_y_true)
            f.write("{:14.8f}{:14.8f}{:14.8f}{:14.8f}{:14.8f}{:14.8f}{:14.8f}{:14.8f}\n".format(rotated_y_pred[0], rotated_y_pred[1], rotated_y_pred[2],
                rotated_y_true[0], rotated_y_true[1], rotated_y_true[2], intensity, true_intensity))


