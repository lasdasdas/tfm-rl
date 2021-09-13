from collections import deque
import sys
import matplotlib.pyplot as plt
import matplotlib
import time
sys.path.append("./../true_environments")
from explorer_pre import Environ
# from explorer_final import Environ
# from snake import Environ



env = Environ()
env.bad_score = env.neutral_score

scores = []
finishes = []
score_mem = deque(maxlen=100)




lim = 200
for j in range(200000):
    state = env.reset()
    score = 0
    for i in range(lim):
        action = env.sample()
        next_state, reward, done, info = env.step(action)
        env.render()
        state = next_state
        score += reward
