from envs.multi_runner import ShareSubprocVecEnv
from generate_road import generate, generate_short
import numpy as np

generate_short(2, 2)
cfgs = ['0011', '1100', '1010', '0101']

env = ShareSubprocVecEnv(4, 2, cfgs)
env.reset()
actions = [np.zeros(4) for _ in range(4)]
step = 0

while True:
    _, _, ifdone, cur_time, _ = env.step(actions)
    step += 1

    print(ifdone, step, cur_time)
    if all(ifdone):
        break

env.close()
