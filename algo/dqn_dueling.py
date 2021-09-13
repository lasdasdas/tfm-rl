import tensorflow
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Input, MaxPooling2D, Lambda, Add, AveragePooling3D, AveragePooling2D, LeakyReLU
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf
import random
from collections import deque
import numpy as np
import sys
import gym
import pickle
import time


sys.path.append("./../environments")
sys.path.append("./../tools")
from explorer_pre import Environ
# from explorer_final import Environ
# from snake import Environ

from per import PER_st as PER


disable_eager_execution()


class Agent():
    def __init__(self, observation_size, action_size):
        self.observation_size = observation_size
        self.action_size = action_size
        self.PER = PER()

        self.batch_size = 512
        self.lr = 1e-5
        self.gamma = 0.99
        self.decay = 5e-4
        self.traincount = 0

        self.create_model()

    def create_model(self):

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

        self.model = Model(inputs=[input], outputs=[Xout])
        self.model.compile(loss=tf.keras.losses.Huber(),
                           optimizer=keras.optimizers.Adam(
            lr=self.lr,  clipnorm=200, decay=self.decay),

            metrics=['MeanSquaredError'])


    def append(self, t):
        self.PER.remember(t)

    def act(self, st):
        st_tmp = st[None, :, :, :]
        ret = np.argmax(self.model.predict(st_tmp, batch_size=1))
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

        target += self.model.predict_on_batch(next_states)

        q_up = rewards + self.gamma * \
            (np.amax(target, axis=1)) \
            * (1 - dones)

        indices = np.array([i for i in range(self.batch_size)])
        q_values = self.model.predict_on_batch(states)
        td = np.abs(q_up - q_values[indices, actions])

        q_values[[indices], [actions]] = q_up
        self.PER.update_indices(per_indices, td + 1)

        history = self.model.fit(states, q_values, epochs=1, verbose=0)


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

for j in range(30000):
    state = env.reset()
    score = 0
    mem = []
    expl *= expl_decay
    for i in range(8*lim):
        action = env.sample()
        if random.uniform(0, 1) > expl:
            action = a.act(state)
        next_state, reward, done, info = env.step(action)
        a.append((state, action, reward, next_state, done))

        state = next_state
        score += reward
        if done or lim - 1 == i:
            a.replay()
            score_mem.append(score)
            print(j,  sum(score_mem) / len(score_mem), expl)
            score = 0
            break
