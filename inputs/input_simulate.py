"""
This file is used to simulate the input from the user.
"""
from utils.utils import get_merge_dictionary

User_Simulate_Input = {
    'algorithm': 'PPO',
    'model_path': 'recur/PPO_64x2_20M_5x5_test.zip',
    'map_path': 'maps/_5x5_twisted1',
    'map_dir': 'maps/generated/all_4X4',
    'render_fps': 1,
}

Default_Simulate_Input = {
    'algorithm': 'PPO',
    'model_path': 'models/PPO/_5X5_2024-03-11_15_07_50/tt_5000_lr_0.0001_bs_64_g_0.99_ns_2048_.zip',
    'map_path': 'maps/_5X5_empty_rotate.txt',
    'map_dir': 'test_result/DQN/success/',
}


def simulate_input():
    return get_merge_dictionary(Default_Simulate_Input, User_Simulate_Input)
