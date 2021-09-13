import copy
import sys
import gym
import pickle
import time
import numpy as np
from collections import deque
import random


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, Input, Concatenate, MaxPooling2D, Lambda, Add, AveragePooling3D, AveragePooling2D, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.python.framework.ops import disable_eager_execution

from tensorflow.keras import backend as K


sys.path.append("./../environments")
sys.path.append("./../tools")
from explorer_pre import Environ
# from explorer_final import Environ
# from snake import Environ

disable_eager_execution()


def toG(r, g=0.9):
    ret = np.zeros_like(r)
    curr = 0
    for t  in reversed(range( len(r))):
        curr = curr * g + r[t]
        ret[t] = curr

    # Baseline
    ret -= np.mean(ret)
    ret /= np.std(ret)
    return ret


class Agent():
    def __init__(self, observation_size, action_size):
        self.observation_size = observation_size
        self.action_size = action_size
        self.mem = []

        self.lr = 1e-5
        self.gamma = 0.9
        self.decay = 5e-8
        self.traincount = 0

        self.value_size = 1
        self.create_model()
        self.loss = deque(maxlen=150)

    def toplayer(self):
        return x, input1, input2, adv_input

    def create_model(self):

        input = Input(shape=self.observation_size, dtype='float32')
        c1 = Conv2D(8, kernel_size=(3, 3), strides=(1, 1),
                    padding='same', activation="relu")(input)
        c1 = Conv2D(16, kernel_size=(1, 1), strides=(1, 1),
                    padding='same', activation="relu")(c1)

        f = Flatten()(c1)

        x = Dense(256)(f)
        x = LeakyReLU(alpha=0.1)(x)

        x = Dense(128)(f)
        x = LeakyReLU(alpha=0.1)(x)

        x = Dense(256)(f)
        x = LeakyReLU(alpha=0.1)(x)

        x = Dense(self.action_size)(x)
        x = Activation('softmax')(x)

        self.model_pg = Model([input], x)
        self.model_pg.compile(
            optimizer=Adam(lr=self.lr),
            loss='categorical_crossentropy',
        )

    def append(self, t):
        self.mem.append(t)

    def act(self, state):
        action_prob = a.model_pg.predict(np.expand_dims(state, axis=0))[0]
        return np.random.choice(len(action_prob), 1, p=action_prob)[0]

    def replay(self, importance=1):

        batch = self.mem

        states = np.array([l[0] for l in batch])
        actions = np.array([l[1] for l in batch])
        rewards = np.array([l[2] for l in batch])
        next_states = np.array([l[3] for l in batch])
        adv = toG(rewards)

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        action_probs = self.model_pg.predict(states)

        acc = np.array(actions)
        indices = np.array([i for i in range(len(batch))]).astype(int)
        actions_oh = np.zeros_like(action_probs)
        actions_oh[[indices], [acc]] = 1

        gradients = actions_oh - action_probs
        discounted_rewards = toG(rewards)
        alpha = 0.0001
        for i in range(len(batch)):
            gradients[i] *= alpha * discounted_rewards[i]
        gradients += action_probs

        history = self.model_pg.fit(states, gradients)
        self.loss.append(history.history["loss"][0])
        loss = history.history["loss"][0]

        self.mem.clear()

        return loss


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
num_steps = 400


for j in range(40000000):
    state = env.reset()
    score = 0
    for i in range(num_steps):

        action = a.act(state)

        next_state, reward, done, info = env.step(action)

        a.append((state, action, reward, next_state, done))
        state = next_state
        score += reward

        if (done or i == num_steps - 1):
            loss = a.replay()
            print("step", j, "score", score, "expl", expl)
            break
