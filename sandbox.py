import numpy as np

from envs.env import FrozenLake
from src.train import make_env
from utils.process_IO import load_map

_map = load_map("maps/_5x5_twisted1")
_desc = np.asarray(_map, dtype='c')
env = make_env(_map=_desc, is_gui=True, truncate=True, random_reset=True)

env.reset()
while True:
    print("=====================================")
    input_action = int(input("Enter action: "))
    if input_action == -1:
        print(env.P)
        continue
    obs, reward, done, truncated, info = env.step(input_action)
    print("obs", obs)
    print("reward", reward)
    print("done", done)
    print("truncated", truncated)
    print("info", info)

    print("=====================================")
    if done:
        reset = env.reset()
        print("reset", reset)
