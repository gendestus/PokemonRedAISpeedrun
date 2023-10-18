from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from argparse_pokemon import *
import sys

if len(sys.argv) < 2:
    print('Usage: eval_model.py <model_path>')
    exit(1)

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init
if __name__ == '__main__':
    model_path = sys.argv[1]

    sess_path = f'eval_{str(uuid.uuid4())[:8]}'
    run_steps = 2048
    args = get_args('run_baseline.py', ep_length=run_steps, sess_path=sess_path)
    env_config = {
                    'headless': False, 'save_final_state': True, 'early_stop': False, 
                    'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': run_steps,
                    'print_rewards': True, 'save_video': True, 'session_path': sess_path,
                    'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, "fast_video": True
                }

    env_config = change_env(env_config, args)
    #eval_env = RedGymEnv(config=env_config)
    num_cpu = 1 #64 #46  # Also sets the number of episodes per training iteration
    eval_env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    eval_callback = EvalCallback(eval_env, best_model_save_path=sess_path, log_path=sess_path, eval_freq=10000, deterministic=True, render=False, n_eval_episodes=1)
    if exists(model_path):
        print('\nloading checkpoint')
        custom_objects = None
        model = PPO.load(model_path, env=eval_env, custom_objects=None)
        model.learn(total_timesteps=0, callback=eval_callback)
