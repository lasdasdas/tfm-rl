import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, Input, Concatenate
from tensorflow.keras.models import Sequential, Model

from collections import deque
from tensorflow.keras import backend as K

from tensorflow.python.framework.ops import disable_eager_execution
import sys
sys.path.append("./../environments")
from explorer_pre import Environ
# from explorer_final import Environ
# from snake import Environ

disable_eager_execution()


class Agent():
    def __init__(self, state_size, action_size):
        self.cr_lr = 0.00001
        self.ac_lr = 0.000004
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        self.gamma = 0.99
        self.build_models()

    def build_models(self):
        input = Input(shape=self.state_size, dtype='float32')
        adv_input = Input(shape=self.value_size)
        c1 = Conv2D(16, kernel_size=(1, 1), strides=(1, 1),
                    padding='same', activation="relu")(input)
        c1 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1),
                    padding='same', activation="relu")(c1)

        f = Flatten()(c1)

        # Critic
        x = Dense(128, activation='relu')(f)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        v_c = Dense(self.value_size, activation='linear')(x)

        cr = Model(inputs=input,  outputs=v_c)
        cr.compile(loss="mse", optimizer=Adam(lr=self.cr_lr),
                   experimental_run_tf_function=False)
        self.cr_model = cr

        # Actor
        x = Dense(128, activation='relu')(f)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        v_a = Dense(self.action_size, activation='softmax')(x)

        def c_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            out2 = K.clip(y_true, 1e-8, 1-1e-8)
            log_lik = out2*K.log(out)
            return K.sum(-log_lik*adv_input)

        ac = Model(inputs=[input, adv_input], outputs=v_a)
        ac.compile(loss=c_loss, optimizer=Adam(
            lr=self.cr_lr), experimental_run_tf_function=False)
        ac.summary()
        self.ac_model = ac

        pol = Model(inputs=input, outputs=v_a)
        self.pol_model = pol

    def act(self, state):
        p = self.pol_model.predict(state[None, :])
        return np.random.choice(self.action_size, 1, p=p[0])[0]


    def replay(self, mem):
        states = np.array([l[0] for l in mem])
        actions = np.array([l[1] for l in mem])
        rewards = np.array([l[2] for l in mem])
        next_states = np.array([l[3] for l in mem])
        ndones = np.invert(np.array([l[4] for l in mem])).astype(np.int)

        values = self.cr_model.predict_on_batch(np.array(states))
        next_values = self.cr_model.predict_on_batch(np.array(next_states))

        next_values = next_values.reshape((len(mem),))
        values = values.reshape((len(mem),))

        delta = rewards + self.gamma*next_values*ndones - values

        target = rewards + self.gamma*next_values*ndones

        acts = np.zeros((len(mem), self.action_size))
        indices = np.array([x for x in range(len(mem))])
        acts[indices, actions] = 1

        self.ac_model.fit([states, delta], acts, verbose=0) 
        self.cr_model.fit(states, rewards, verbose=0)


env = Environ()
a = Agent((env.board_size, env.board_size, 3), env.num_actions)

# Snake needs 4 channels
# a = Agent((env.board_size, env.board_size, 4), env.num_actions)

scores = []
score_mem = deque(maxlen=100)
for j in range(30000):
    state = env.reset()
    score = 0
    mem = []
    lim = 200
    for i in range(lim):
        action = a.act(state)
        next_state, reward, done, info = env.step(action)
        mem.append((state, action, reward, next_state, done))

        state = next_state
        score += reward

        if done or lim - 1 == i:
            a.replay(mem)
            mem.clear()
            score_mem.append(score)
            print(j, score, sum(score_mem) / len(score_mem))
            score = 0
            break
    state = env.reset()

env.close()
