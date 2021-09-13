from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, Input, Concatenate, MaxPooling2D, Lambda, Add, AveragePooling3D, AveragePooling2D, LeakyReLU
from tensorflow import keras
import tensorflow as tf
import argparse
import os.path
import matplotlib.pyplot as plt
import gym
import pickle
import time
import numpy as np
from collections import deque
import random

import sys
sys.path.append("./../environments")
sys.path.append("./../tools")
from explorer_pre import Environ
# from explorer_final import Environ
# from snake import Environ
from per import PER_st as PER


tf.compat.v1.disable_eager_execution()


class Agent():
    def __init__(self, observation_size, action_size):
        self.observation_size = observation_size
        self.action_size = action_size
        persize = 17
        self.lr = 1e-5

        self.PER = PER(persize)
        # Inverse unsample
        # self.PER = PER(persize, 1, False)

        self.batch_size = 256
        self.gamma = 0.95
        self.decay = 3e-4
        self.traincount = 0
        self.create_model()


    def create_model(self):
        self.k = 5
        self.k_roll = 1

        self.model_queue = deque(maxlen=self.k)
        for i in range(self.k):
            input = Input(shape=self.observation_size, dtype='float32')
            c1 = Conv2D(8, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', activation="relu")(input)
            c1 = Conv2D(16, kernel_size=(1, 1), strides=(1, 1),
                        padding='same', activation="relu")(c1)

            f = Flatten()(c1)

            x = Dense(128)(f)
            x = LeakyReLU(alpha=0.1)(x)
            x = Dense(256)(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = Dense(128)(x)
            x = LeakyReLU(alpha=0.1)(x)

            xv = Dense(1)(x)
            xa = Dense(self.action_size)(x)
            Xout = (xv + (xa - tf.math.reduce_mean(xa, axis=1, keepdims=True)))

            model = Model(inputs=[input], outputs=[Xout])
            if i == 0:
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    self.lr,
                    decay_steps=1000000,
                    decay_rate=self.decay,
                    staircase=True)
                model.compile(loss=tf.keras.losses.Huber(),
                              optimizer=keras.optimizers.Adam(
                    clipnorm=20, learning_rate=lr_schedule),
                    metrics=['MeanSquaredError'])
            self.model_queue.append((model))

    def append(self, t):
        self.PER.remember(t)

    def act(self, st):
        st_tmp = st[None, :, :, :]
        i = random.randint(0, self.k - 1)
        ret = np.argmax(self.model_queue[i].predict(st_tmp, batch_size=1))
        return ret

    def replay(self):
        if len(self.PER) < 1:
            return

        batch, per_indices = self.PER.sample(self.batch_size)
        states = np.array([l[0] for l in batch])
        actions = np.array([l[1] for l in batch])
        rewards = np.array([l[2] for l in batch])
        next_states = np.array([l[3] for l in batch])
        dones = np.array([l[4] for l in batch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        target = np.zeros((self.batch_size, self.action_size))
        for i in range(self.k):
            target += self.model_queue[i].predict_on_batch(next_states)
        target /= self.k

        q_up = rewards + self.gamma * \
            (np.amax(target, axis=1)) \
            * (1 - dones)

        indices = np.array([i for i in range(self.batch_size)])
        q_values = self.model_queue[0].predict_on_batch(states)
        td = np.abs(q_up - q_values[indices, actions])
        q_values[[indices], [actions]] = q_up

        self.PER.update_indices(per_indices, td + 0.001)

        if self.traincount % 300 == 0:
            print("train steps", self.traincount)
            # self.model_queue[0].save_weights('model_weights.h5')
            self.model_queue[self.k_roll].set_weights(
                self.model_queue[0].get_weights())
            self.k_roll += 1
            if self.k_roll == self.k:
                self.k_roll = 1

        self.traincount += 1
        history = self.model_queue[0].fit(
            states, q_values, epochs=1, verbose=0)


scores = deque(maxlen=450)
scores_frac = deque(maxlen=450)
scores_frac_g = deque(maxlen=450)
expl_decay = 0.9999
min_expl = 0.00000001
expl = 1
num_steps = 400


env = Environ()

a = Agent((env.board_size, env.board_size, 3), env.action_size)
# Snake needs 4 channels
# a = Agent((env.board_size, env.board_size, 4), env.num_actions)

scores = deque(maxlen=450)
scores_frac = deque(maxlen=450)
scores_frac_g = deque(maxlen=450)
expl_decay = 0.9999
min_expl = 0.00000001
expl = 1
lim = 200


scores = []
finishes = []
score_mem = deque(maxlen=100)


for j in range(100000):
    state = env.reset()
    score = 0
    mem = []
    for i in range(lim):
        action = env.sample()
        if random.uniform(0, 1) > expl:
            action = a.act(state)
        next_state, reward, done, info = env.step(action)
        a.append((state, action, reward, next_state, done))

        state = next_state
        score += reward
        if j % 150 == 0:
            env.render()
        if i % 50 == 0:
            a.replay()

        if done or lim - 1 == i:
            expl = max(min_expl, expl_decay * expl)
            a.replay()
            score_mem.append(score)
            print(j,   sum(score_mem) / len(score_mem), expl)
            score = 0
            break

    state = env.reset()
