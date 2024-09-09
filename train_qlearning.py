import envs
from pathlib import Path
import random
import time
from matplotlib import pyplot as plt
import mo_gymnasium as mo_gym
from motndp.city import City
from motndp.constraints import MetroConstraints
import numpy as np
from qlearning_tndp import QLearningTNDP
# import torch
import argparse
import wandb

def main(args):
    def make_env(gym_env):
        city = City(
            args.city_path,
            groups_file=args.groups_file,
            ignore_existing_lines=args.ignore_existing_lines
        )
        
        env = mo_gym.make(gym_env, 
                        city=city, 
                        constraints=MetroConstraints(city),
                        nr_stations=args.nr_stations,
                        od_type=args.od_type,
                        chained_reward=args.chained_reward,)

        return env
    
    if args.evaluate_model is not None:
        api = wandb.Api()

        run = api.run(f"TNDP-RL/{args.evaluate_model}")
        run.file(f"q_tables/{args.evaluate_model}.npy").download(replace=True)
        
        import json
        run_config = json.loads(run.json_config)
        
        city = City(
            args.city_path,
            groups_file=args.groups_file,
            ignore_existing_lines=run_config['ignore_existing_lines']['value']
        )
        
        env = mo_gym.make(run_config['env_id']['value'], 
                        city=city, 
                        constraints=MetroConstraints(city),
                        nr_stations=args.nr_stations,
                        od_type=run_config['od_type']['value'],
                        chained_reward=run_config['chained_reward']['value'],)
        
        # Load the Q-table  
        Q = np.load(f"q_tables/{args.evaluate_model}.npy")
        agent = QLearningTNDP(
            env,
            alpha=args.alpha,
            gamma=args.gamma,
            initial_epsilon=args.initial_epsilon,
            final_epsilon=args.final_epsilon,
            epsilon_decay_steps=args.epsilon_decay_steps,
            train_episodes=args.train_episodes,
            test_episodes=args.test_episodes,
            nr_stations=args.nr_stations,
            policy=None,
            seed=args.seed,
            wandb_project_name=args.project_name,
            wandb_experiment_name=args.experiment_name,
            wandb_run_id=args.evaluate_model,
            Q_table=Q,
        )
        
        agent.test(args.test_episodes, starting_loc=args.starting_loc)

    else:
        env = make_env(args.gym_env)
        agent = QLearningTNDP(
            env,
            alpha=args.alpha,
            gamma=args.gamma,
            initial_epsilon=args.initial_epsilon,
            final_epsilon=args.final_epsilon,
            epsilon_decay_steps=args.epsilon_decay_steps,
            train_episodes=args.train_episodes,
            test_episodes=args.test_episodes,
            nr_stations=args.nr_stations,
            policy=args.policy,
            seed=args.seed,
            wandb_project_name=args.project_name,
            wandb_experiment_name=args.experiment_name
        )
        agent.train(args.starting_loc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabular Q-learning for MO-TNDP")
    # Acceptable values: 'dilemma', 'margins', 'amsterdam'
    parser.add_argument('--env', default='dilemma', type=str)
    # For xian/amsterdam environment we have different groups files (different nr of objectives)
    parser.add_argument('--nr_groups', default=5, type=int)
    # Starting location of the agent
    parser.add_argument('--starting_loc_x', default=None, type=int)
    parser.add_argument('--starting_loc_y', default=None, type=int)
    parser.add_argument('--nr_stations', type=int)
    parser.add_argument('--policy', default=None, type=str, help="WARNING: set manually, does not currently convert string to list. A list of action as manual policy for the agent.")
    parser.add_argument('--alpha', default=0.4, type=float)
    parser.add_argument('--gamma', default=0.8, type=float)
    parser.add_argument('--initial_epsilon', default=1.0, type=float)
    parser.add_argument('--final_epsilon', default=0.0, type=float)
    parser.add_argument('--epsilon_decay_steps', default=400, type=float)
    parser.add_argument('--train_episodes', default=500, type=int)
    parser.add_argument('--test_episodes', default=1, type=int)
    parser.add_argument('--no_log', action='store_true', default=False)
    parser.add_argument('--ignore_existing_lines', action='store_true', default=False)
    parser.add_argument('--od_type', default='pct', type=str, choices=['pct', 'abs'])
    parser.add_argument('--chained_reward', action='store_true', default=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--evaluate_model', default=None, type=str, help="Wandb run ID for model to evaluate. Will load the Q table and run --test_episodes. Note that starting_loc will be set to the one with the max Q.") 

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    Path("./q_tables").mkdir(parents=True, exist_ok=True)
    
    args.project_name = "TNDP-RL"

    # Some values are hardcoded for each environment (this is flexible, but we don't want to have to pass 100 arguments to the script)
    if args.env == 'dilemma':
        args.city_path = Path(f"./envs/mo-tndp/cities/dilemma_5x5")
        args.nr_stations = 9
        args.gym_env = 'motndp_dilemma-v0'
        args.groups_file = "groups.txt"
        args.ignore_existing_lines = args.ignore_existing_lines
        args.experiment_name = "Q-Learning-Dilemma"
    elif args.env == 'margins':
        args.city_path = Path(f"./envs/mo-tndp/cities/margins_5x5")
        args.nr_stations = 9
        args.gym_env = 'motndp_margins-v0'
        args.groups_file = f"groups.txt"
        args.ignore_existing_lines = args.ignore_existing_lines
        args.experiment_name = "Q-Learning-Margins"
    elif args.env == 'amsterdam':
        args.city_path = Path(f"./envs/mo-tndp/cities/amsterdam")
        args.nr_stations = args.nr_stations
        args.gym_env = 'motndp_amsterdam-v0'
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.ignore_existing_lines = args.ignore_existing_lines
        args.experiment_name = "Q-Learning-Amsterdam"
    elif args.env == 'xian':
        # Xian pre-defined
        args.city_path = Path(f"./envs/mo-tndp/cities/xian")
        args.nr_stations = 45
        args.gym_env = 'motndp_xian-v0'
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.ignore_existing_lines = args.ignore_existing_lines
        args.experiment_name = "Q-Learning-Xian"
        # args.od_type = 'abs'
        # args.policy = [5, 5, 5, 6, 6, 6, 6, 6, 4, 6, 4, 4, 6, 6, 6, 6, 6, 6, 4, 4, 4, 6, 4, 6, 6, 6, 6, 4, 4, 6, 6, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6]
        # args.policy = [0, 2, 1, 0, 0, 0, 6, 6, 0, 0, 0, 0, 6, 6, 0, 0, 6, 0, 6, 6, 4, 6, 6, 4, 4, 4, 5, 6, 6, 4, 4, 6, 4, 6, 6, 4, 4, 3, 2, 3, 2, 4, 2, 2]
        # args.policy = [2, 4, 2, 2, 0, 0, 1, 2, 2, 1, 0, 2, 0, 0, 6, 6, 5, 5, 6, 4, 6, 6, 4, 6, 4, 6, 4, 4, 6, 6, 6, 7, 6, 0, 0, 1, 2, 1, 2, 0, 0, 6, 6, 0]
        # if args.policy is not None:
        #     args.starting_loc_x = 19
        #     args.starting_loc_y = 11

    if args.starting_loc_x is not None and args.starting_loc_y is not None:
        args.starting_loc = (args.starting_loc_x, args.starting_loc_y)
    else:
        args.starting_loc = None

    main(args)
