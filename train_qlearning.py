from pathlib import Path
import random
import time
from matplotlib import pyplot as plt
import mo_gymnasium as mo_gym
from motndp.city import City
from motndp.constraints import MetroConstraints
import numpy as np
import torch
import envs
import argparse
import wandb

from morl_baselines.common.utils import linearly_decaying_value

class QLearningTNDP:
    def __init__(
        self,
        env,
        alpha: float,
        gamma: float,
        initial_epsilon,
        final_epsilon,
        epsilon_decay_steps,
        train_episodes,
        test_episodes,
        nr_stations,
        seed,
        policy = None,
        wandb_project_name=None,
        wandb_experiment_name=None,
        wandb_run_id=None,
        Q_table=None,
        log: bool = True
    ):
        self.env = env
        self.env_id = env.unwrapped.spec.id
        self.alpha = alpha
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.train_episodes = train_episodes
        self.test_episodes = test_episodes
        self.nr_stations = nr_stations
        self.seed = seed
        self.policy = policy
        self.wandb_project_name = wandb_project_name
        self.wandb_experiment_name = wandb_experiment_name
        self.wandb_run_id = wandb_run_id
        if Q_table is not None:
            self.Q = Q_table # Q_table to start with or evaluate
        self.log = log
        
        if log:
            if not wandb_run_id:
                self.setup_wandb()
            else:
                wandb.init(
                    project=self.wandb_project_name,
                    id=self.wandb_run_id,
                    resume=True,
                )
        
    def get_config(self) -> dict:
        """Get configuration of QLearning."""
        return {
            "env_id": self.env_id,
            "od_type": self.env.od_type,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "final_epsilon": self.final_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "train_episodes": self.train_episodes,
            "test_episodes": self.test_episodes,
            "nr_stations": self.nr_stations,
            "seed": self.seed,
            "policy": self.policy,
            "chained_reward": self.env.chained_reward,
            "ignore_existing_lines": self.env.city.ignore_existing_lines,
        }
        
    def highlight_cells(self, cells, ax, **kwargs):
        """Highlights a cell in a grid plot. https://stackoverflow.com/questions/56654952/how-to-mark-cells-in-matplotlib-pyplot-imshow-drawing-cell-borders
        """
        for cell in cells:
            (y, x) = cell
            rect = plt.Rectangle((x-.5, y-.5), 1, 1, fill=False, linewidth=2, **kwargs)
            ax.add_patch(rect)
        return rect
    
    def gen_line_plot_grid(self, lines):
        """Generates a grid_x_max * grid_y_max grid where each grid is valued by the frequency it appears in the generated lines.
        Essentially creates a grid of the given line to plot later on.

        Args:
            line (list): list of generated lines of the model
            grid_x_max (int): nr of lines in the grid
            grid_y_mask (int): nr of columns in the grid
        """
        data = np.zeros((self.env.city.grid_x_size, self.env.city.grid_y_size))

        for line in lines:
            # line_g = city.vector_to_grid(line)

            for station in line:
                data[station[0], station[1]] += 1
        
        data = data/len(lines)

        return data

    
    def setup_wandb(self, entity=None, group=None):
        wandb.init(
            project=self.wandb_project_name,
            entity=entity,
            config=self.get_config(),
            name=f"{self.env_id}__{self.wandb_experiment_name}__{self.seed}__{int(time.time())}",
            save_code=True,
            group=group,
        )
        
        wandb.define_metric("*", step_metric="episode")


    def train(self, starting_loc=None):
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        
        rewards = []
        avg_rewards = []
        epsilons = []
        training_step = 0
        best_episode_reward = 0
        best_episode_segment = []
        # All the ACTUALIZED starting locations of the agent.
        actual_starting_locs = set()
        epsilon = self.initial_epsilon
        
        for episode in range(self.train_episodes):
            # Initialize starting location
            if starting_loc:
                if type(starting_loc[0]) == tuple:
                    loc = (random.randint(*starting_loc[0]), random.randint(*starting_loc[1]))
                else:
                    loc = starting_loc
            else:
                # Set the starting loc via e-greedy policy
                exp_exp_tradeoff = random.uniform(0, 1)
                # exploit
                if exp_exp_tradeoff > epsilon:
                    loc = tuple(self.env.city.vector_to_grid(np.unravel_index(self.Q.argmax(), self.Q.shape)[0]))
                # explore
                else:
                    loc = None

            if episode == 0:
                state, info = self.env.reset(seed=self.seed, loc=loc)
            else:
                state, info = self.env.reset(loc=loc)

            actual_starting_locs.add((state['location'][0], state['location'][1]))
            episode_reward = 0
            episode_step = 0
            while True:
                state_index = self.env.city.grid_to_vector(state['location'][None, :]).item()

                # Exploration-exploitation trade-off
                exp_exp_tradeoff = random.uniform(0, 1)

                # follow predetermined policy (set above)
                if self.policy:
                    action = self.policy[episode_step]
                # exploit
                elif exp_exp_tradeoff > epsilon:
                    action = np.argmax(self.Q[state_index, :] - 10000000 * (1-info['action_mask']))
                # explore
                else:
                    action = self.env.action_space.sample(mask=info['action_mask'])
                
                new_state, reward, done, _, info = self.env.step(action)

                # Here we sum the reward to create a single-objective policy optimization
                reward = reward.sum()

                # Update Q-Table
                new_state_gid = self.env.city.grid_to_vector(new_state['location'][None, :]).item()
                self.Q[state_index, action] = self.Q[state_index, action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state_gid, :]) - self.Q[state_index, action])
                episode_reward += reward

                training_step += 1
                episode_step += 1

                state = new_state

                if done:
                    print('segments:', self.env.all_sat_od_pairs)
                    print('line', self.env.covered_cells_vid)
                    break
            
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                best_episode_segment = info['segments']
            
            # Adding the total reward and reduced epsilon values
            rewards.append(episode_reward)
            # Save the average reward over the last 10 episodes
            avg_rewards.append(np.average(rewards[-10:]))
            epsilons.append(epsilon)
            
            if self.log:
                wandb.log(
                    {
                        "episode": episode,
                        "reward": episode_reward,
                        "average_reward": avg_rewards[-1],
                        "training_step": training_step,
                        "epsilon": epsilon,
                        "best_episode_reward": best_episode_reward,
                    })

            print(f'episode: {episode}, reward: {episode_reward} average rewards of last 10 episodes: {avg_rewards[-1]}')
            
            #Cutting down on exploration by reducing the epsilon
            epsilon = linearly_decaying_value(self.initial_epsilon, self.epsilon_decay_steps, episode, 0, self.final_epsilon)
        
        if self.log:
            # Log the final Q-table
            final_Q_table = Path(f"./q_tables/{wandb.run.id}.npy")
            np.save(final_Q_table, self.Q)
            wandb.save(final_Q_table.as_posix())
            
            # Log the Q-table as an image
            fig, ax = plt.subplots(figsize=(10, 5))
            Q_actions = self.Q.argmax(axis=1).reshape(self.env.city.grid_x_size, self.env.city.grid_y_size)
            Q_values = self.Q.max(axis=1).reshape(self.env.city.grid_x_size, self.env.city.grid_y_size)
            im = ax.imshow(Q_values, label='Q values', cmap='Blues', alpha=0.5)
            markers = ['\\uparrow', '\\nearrow', '\\rightarrow', '\\searrow', '\\downarrow', '\\swarrow', '\\leftarrow', '\\nwarrow']
            for a in range(8):
                cells = np.nonzero((Q_actions == a) & (Q_values > 0))
                ax.scatter(cells[1], cells[0], c='red', marker=rf"${markers[a]}$", s=10,)
            
            fig.colorbar(im)
            fig.suptitle('Q values and best actions')
            self.highlight_cells(actual_starting_locs, ax=ax, color='limegreen')
            wandb.log({"Q-table": wandb.Image(fig)})
            plt.close(fig)
            
            if self.test_episodes > 0:
                self.test(self.test_episodes, starting_loc)
        
            wandb.finish()
        return self.Q, rewards, avg_rewards, epsilons, best_episode_reward, best_episode_segment, actual_starting_locs


    def test(self, test_episodes, starting_loc=None):
        total_rewards = 0
        generated_lines = []
        if starting_loc:
            test_starting_loc = starting_loc
        else:
            test_starting_loc = tuple(self.env.city.vector_to_grid(np.unravel_index(self.Q.argmax(), self.Q.shape)[0]))
            
        for episode in range(test_episodes):
            state, info = self.env.reset(loc=test_starting_loc)
            locations = [state['location'].tolist()]
            actions = []
            episode_reward = 0
            while True:
                state_index = self.env.city.grid_to_vector(state['location'][None, :]).item()
                action = np.argmax(self.Q[state_index, :] - 10000000 * (1-info['action_mask']))
                actions.append(action)
                new_state, reward, done, _, info = self.env.step(action)
                locations.append(new_state['location'].tolist())
                reward = reward.sum()
                episode_reward += reward
                state = new_state
                if done:
                    break
            total_rewards += episode_reward
            generated_lines.append(locations)
            
        if self.log:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(self.env.city.agg_od_mx())
            
            if not args.ignore_existing_lines:
                for i, l in enumerate(self.env.city.existing_lines):
                    station_locs = self.env.city.vector_to_grid(l)
                    ax.plot(station_locs[:, 1], station_locs[:, 0], '-o', color='#A1A9FF', label='Existing lines' if i == 0 else None)
            
            # If the test episodes are only 1, we can plot the line directly, with connected points
            if len(generated_lines) == 1:
                station_locs = np.array(generated_lines[0])
                ax.plot(station_locs[:, 1], station_locs[:, 0], '-or', label='Generated line')
            # If the test episodes are more than 1, we can plot the average line
            else:
                plot_grid = self.gen_line_plot_grid(np.array(generated_lines))
                station_locs = plot_grid.nonzero()
                ax.plot(station_locs[1], station_locs[0], 'ok', label='Generated line')

            self.highlight_cells([test_starting_loc], ax=ax, color='limegreen')
            fig.suptitle(f'Average Generated line \n reward: {episode_reward}')
            fig.legend(loc='lower center', ncol=2)
            wandb.log({"Average-Generated-Line": wandb.Image(fig)})
            plt.close(fig)
            
        print(f'Average reward over {test_episodes} episodes: {total_rewards/test_episodes}')
        print(f'Actions of last episode: {actions}')

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
    torch.manual_seed(args.seed)
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
