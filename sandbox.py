from envs.my_frozen_lake import FrozenLakeEnv
from src.train import make_env
from utils.process_IO import load_map
from utils.utils import generate_random_map

_map = load_map("maps/_5x5_twisted1")
env = make_env(_map, is_gui=True, truncate=False, random_reset=False)
#
# print(generate_random_map(5, 5, 0.5))
#
# print(_map)
#
# def to_boolean(x):
#     return False if x == b'H' else True
#
#
# grid = map(to_boolean, grid)
# print(list(grid))


print(env.reset())
while True:
    print("=====================================")
    input_action = int(input("Enter action: "))
    if input_action == -1:
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

