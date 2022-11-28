import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras_tuner as kt
import os



def split_data(df, train_p, valid_p):
    train_len = round(train_p * len(df))
    valid_len = round(valid_p * len(df))
    test_len = len(df) - train_len - valid_len

    df_train = df[:train_len]
    df_valid = df[train_len:train_len + valid_len]
    df_test = df[-test_len:]

    return df_train, df_valid, df_test




def load_data(df, max_depth, sampling_rate):

    X=[]
    Y=[]
    num_y = int(max_depth/sampling_rate)

    for i in range(len(df)):
        x = np.array([df['Latitude'][i], df['Longitude'][i]])
        y = np.array([float(y) for y in df['Speed Array'][0].split(' ')])
        if len(y)>= num_y:
            X.append(x)
            Y.append(y[:num_y])
    return np.array(X), np.array(Y)





class MyHyperModel(kt.HyperModel):

    """
    Class that defines the architecture of the feed forward neural network

    variables:
        x_shape, y_shape: shape of the input and output of the model
        model: a hypermodel for the feed forward neural network
    """

    def __init__(self, x_shape, y_shape):
        self.x_shape = x_shape
        self.y_shape = y_shape

    def build(self, hp):

        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(self.x_shape[-1],)))
        for i in range(hp.Int('n_layers',1,3)):
            model.add(tf.keras.layers.Dense(units= hp.Int('units_'+str(i),min_value= 28, max_value=528, step=100), activation='relu'))
            model.add(tf.keras.layers.Dropout(hp.Float('Dropout_rate'+str(i), min_value=0.1, max_value= 0.4, step=0.1)))
        model.add(tf.keras.layers.Dense(self.y_shape[-1]))
        model.compile(loss='mae',
                      optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate',
                                                                   values=[1e-2,1e-4])),metrics =['accuracy'])

        return model


def optimize_hyperparams(dir, train_tuple,valid_tuple,epochs=2):

    """
    Optimizes hyperparameters for the feed forward neural network

    arguments:
        dir: directory for storing data
        train_tuple, valid_tuple: input and output data for training and validation
        epochs: number of epochs ran in the training

    return:
        model, best_hp: best model and hyperparameters from the search

    additional variables:
        hyper_model: feed forward neural network model for hyperparameter optimization
        tuner: object for search over possible hyperparameter combinations
    """

    # Initiates the directory if it is not already made
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Defines an object of the hypermodel
    hyper_model = MyHyperModel(x_shape=train_tuple[0].shape, y_shape=train_tuple[1].shape)

    # Does 20 random combinations of the hyperparameters given in the mlf.MyHyperModels to find the best one
    tuner = kt.tuners.RandomSearch(hypermodel=hyper_model,objective="val_loss", max_trials=5,
                                   directory=dir,project_name='best parameters')
    tuner.executions_per_trial = 1  # For each combination, 3 executions are ran to prevent mistakes
    tuner.search(train_tuple[0],train_tuple[1], epochs= epochs, validation_data=(valid_tuple[0],valid_tuple[1]))

    # Returns the best parameters from the search and the best model
    best_hp = tuner.get_best_hyperparameters()[0]
    model = tuner.get_best_models()[0]

    return model, best_hp


def error_calc(pred_vals, targets, target_tags, data_type):

    """
    Evaluates the mean absolute error between the predictive model and actual values

    arguments:
        pred_vals: array with values predicted by the model
        targets: array with actual values to compare with the predicted values
        target_tags: List of tags for targets

    return:
        dict: dictionary with different errors

    additional variables:
        err, avg_err: the mean absolute error between the predictions and the actual values
        df_err: DataFrame with err and avg_err for evaluation
    """

    # Calculate the mean absolute error between vectors/matrices
    err = np.mean(np.abs(targets - pred_vals), axis=0)
    err = [round(num,2) for num in err]
    avg_err = np.mean(err)


    return avg_err