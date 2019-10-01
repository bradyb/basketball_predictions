from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

game_data_files_list = ['2013.pkl', '2014.pkl', '2015.pkl', '2016.pkl',
        '2017.pkl', '2018.pkl', '2019.pkl']

df_list = []
for game_data_file in game_data_files_list:
    df_list.append(pickle.load(open('../data/' + game_data_file, 'rb')))

raw_data = pd.concat(df_list)
print(raw_data)

# Cleaning
raw_data['effective_field_goal_percentage'] = (raw_data['away_effective_field_goal_percentage']
    - raw_data['home_effective_field_goal_percentage'])
raw_data['free_throw_percentage'] = (raw_data['away_free_throw_percentage']
        - raw_data['home_free_throw_percentage'])
raw_data['offensive_rebounds'] = (raw_data['away_offensive_rebounds']
        - raw_data['home_offensive_rebounds'])
raw_data['assists'] = raw_data['away_assists'] - raw_data['home_assists']
raw_data['turnovers'] = raw_data['away_turnovers'] - raw_data['home_turnovers']
raw_data['free_throw_attempts'] = (raw_data['away_free_throw_attempts']
    - raw_data['home_free_throw_attempts'])
raw_data['offensive_rating'] = raw_data['away_offensive_rating'] - raw_data['home_offensive_rating']
raw_data['point_spread'] = raw_data['away_points'] - raw_data['home_points']
raw_data = raw_data.dropna()
data = raw_data[['effective_field_goal_percentage',       
                 'free_throw_percentage',
                 'offensive_rebounds',
                 'assists',
                 'turnovers',
                 'free_throw_attempts',
                 'offensive_rating',
                 'point_spread']]

print(data)

def normalize(x, stats):
    return (x - stats['mean']) / stats['std']

train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)

train_stats = train_data.describe()
train_stats.pop('point_spread')
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_data.pop('point_spread')
test_labels = test_data.pop('point_spread')

normed_train_data = normalize(train_data, train_stats)
normed_test_data = normalize(test_data, train_stats)
print(normed_train_data)

def build_model(keys):
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(keys)]),
        layers.Dense(64, activation=tf.nn.relu), layers.Dense(1)])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model(train_data.keys())

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')

EPOCHS = 100

history = model.fit(normed_train_data,
                    train_labels,
                    epochs=EPOCHS,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()
#plot_history(history)
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Test mean abs error: {}".format(mae))
