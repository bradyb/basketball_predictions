from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
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
raw_data['field_goal_percentage'] = (raw_data['away_field_goal_percentage']
    - raw_data['home_field_goal_percentage'])
raw_data['three_point_field_goal_percentage'] = (raw_data['away_three_point_field_goal_percentage']
        - raw_data['home_three_point_field_goal_percentage'])
raw_data['free_throw_percentage'] = (raw_data['away_free_throw_percentage']
        - raw_data['home_free_throw_percentage'])
raw_data['offensive_rebounds'] = (raw_data['away_offensive_rebounds']
        - raw_data['home_offensive_rebounds'])
raw_data['assists'] = raw_data['away_assists'] - raw_data['home_assists']
raw_data['turnovers'] = raw_data['away_turnovers'] - raw_data['home_turnovers']
raw_data['free_throw_attempts'] = (raw_data['away_free_throw_attempts']
    - raw_data['home_free_throw_attempts'])
raw_data['point_spread'] = raw_data['away_points'] - raw_data['home_points']
raw_data = raw_data.dropna()
data = raw_data[['field_goal_percentage',
                 'three_point_field_goal_percentage',
                 'free_throw_percentage',
                 'offensive_rebounds',
                 'assists',
                 'turnovers',
                 'free_throw_attempts',
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


