"""
Description : This file contains the default input for the model, you can edit the input as per your requirements.
"""
from utils.utils import get_merge_dictionary

User_Evaluate_Input = {
    'algorithm': 'A2C',
    'model_path': 'network48/A2C_64x4_100M_pn00.zip',
    'n_eval_episodes': 5,
    'map_dir': 'maps/generated/all_4X4',
    'success_map_target': ''
}

Default_Evaluate_Input = {
    'algorithm': 'PPO',
    'model_path': 'models/PPO/_5X5_2024-03-11_15_07_50/tt_5000_lr_0.0001_bs_64_g_0.99_ns_2048_.zip',
    'map_path': 'maps/_5X5_normal.txt',
    'n_eval_episodes': 1,
    'map_dir': 'maps/generated/all_4X4',
    'success_map_target': 'maps/success'
}


def evaluate_input():
    return get_merge_dictionary(Default_Evaluate_Input, User_Evaluate_Input)
