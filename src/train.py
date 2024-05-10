import gymnasium
import numpy as np

from envs.env import FrozenLake
from inputs.env_setting import env_setting
from inputs.input_train import train_input
from src.model import get_algorithm, prune_hyperparameters
from utils.process_IO import create_directory_if_not_exists, get_model_name, get_model_path, \
    get_log_path, get_map_name, load_map
from utils.utils import generate_random_map


def train_model(
        env: gymnasium.Env = None,
        algorithm: str = "PPO",
        model_target: str = "",
        model_name: str = "new_model",
        hyperparameters: dict = None,
        log_target: str = "",
        save: bool = True,
):
    """
    train the model using the given algorithm and env
    then save the model with the given name
    :param env: gym environment
    :param algorithm: algorithm to use for training
    :param model_target: path to the directory to save the model
    :param model_name: name of the model to save
    :param hyperparameters:
    :param log_target:
    :param save: save the model or not
    :return: trained model
    """

    timesteps = hyperparameters['total_timesteps']

    agent_hyperparameters = hyperparameters
    agent_hyperparameters.pop('total_timesteps')

    make_model = get_algorithm(algorithm)

    agent_hyperparameters = prune_hyperparameters(hyperparameters, algorithm)

    model = make_model("MultiInputPolicy", env, verbose=1, tensorboard_log=log_target, **agent_hyperparameters)
    # print(model.policy_kwargs)
    print(f"env : {env.observation_space}")
    print(model.policy)
    model.learn(total_timesteps=timesteps)
    if save:
        model.save(f"{model_target}/{model_name}")
        print(f"Model saved at {model_target}/{model_name}")
    print(f"log saved at {log_target}")
    return model


def make_env(_map=None, is_gui=False, truncate=False, random_reset=False):
    if is_gui:
        render_mode = 'human'
    else:
        render_mode = None

    _env_setting = env_setting()
    print("_map", _map)
    if _map is None:
        map_desc=generate_random_map(4, 4, 0.5)
        return FrozenLake(desc=map_desc,
                          is_slippery=False,
                          render_mode=render_mode,
                          hole_penalty=_env_setting['hole_penalty'],
                          random_reset=random_reset,
                          frozen_prob=_env_setting['frozen_prob'],
                          render_fps=_env_setting['render_fps'],
                          truncate=truncate)
    else:
        map_desc = np.asarray(_map, dtype='c')
        return FrozenLake(desc=map_desc,
                          is_slippery=False,
                          render_mode=render_mode,
                          hole_penalty=_env_setting['hole_penalty'],
                          random_reset=random_reset,
                          frozen_prob=_env_setting['frozen_prob'],
                          render_fps=_env_setting['render_fps'],
                          truncate=truncate)


def train_command():
    env_options = train_input()
    print("detected env options", env_options)
    algorithm = env_options['algorithm']['name']
    hyperparameters = env_options['algorithm']['hyperparameters']

    _map = load_map(env_options['map_path'])
    print("loaded map for train", _map)

    # make environment
    env = make_env(_map=_map, is_gui=False, random_reset=True)

    # process model name and log name
    map_name = get_map_name(env_options['map_path'], env_options['map_size'])
    model_name = get_model_name(env_options['model_name'], hyperparameters)
    model_dir = get_model_path(algorithm, env_options['model_target'], map_name)
    log_dir = get_log_path(algorithm, env_options['log_target'], map_name)

    # if directory does not exist then create it
    create_directory_if_not_exists(model_dir)
    create_directory_if_not_exists(log_dir)

    train_model(env=env, algorithm=algorithm, model_target=model_dir, model_name=model_name,
                hyperparameters=hyperparameters, log_target=log_dir)
