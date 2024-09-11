import wandb
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from pathlib import Path

def linearly_decaying_value(initial_value, decay_period, step, warmup_steps, final_value):
    # Got it from https://github.com/LucasAlegre/morl-baselines
    """Returns the current value for a linearly decaying parameter.

    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.

    Args:
        decay_period: float, the period over which the value is decayed.
        step: int, the number of training steps completed so far.
        warmup_steps: int, the number of steps taken before the value is decayed.
        final value: float, the final value to which to decay the value parameter.

    Returns:
        A float, the current value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (initial_value - final_value) * steps_left / decay_period
    value = final_value + bonus
    value = np.clip(value, min(initial_value, final_value), max(initial_value, final_value))
    return value


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
                    config=self.get_config(),
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


    def train(self, reward_type, starting_loc=None):
        wandb.config['reward_type'] = reward_type
        
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        
        rewards = []
        avg_rewards = []
        epsilons = []
        training_step = 0
        best_episode_reward = 0
        best_episode_cells = []
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
                    action = np.argmax(self.Q[state_index, :] - 10000000 * (1-info['action_mask'].astype(np.int64)))
                # explore
                else:
                    action = self.env.action_space.sample(mask=info['action_mask'])

                new_state, reward, done, _, info = self.env.step(action)

                # Here we sum the reward to create a single-objective policy optimization
                if reward_type == 'max_efficiency':
                    reward = reward.sum()
                else:
                    raise ValueError(f"Reward type {reward_type} not implemented")
                        
                # Update Q-Table
                new_state_gid = self.env.city.grid_to_vector(new_state['location'][None, :]).item()
                self.Q[state_index, action] = self.Q[state_index, action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state_gid, :]) - self.Q[state_index, action])
                episode_reward += reward

                training_step += 1
                episode_step += 1

                state = new_state

                if done:
                    # print('segments:', self.env.all_sat_od_pairs)
                    # print('line', self.env.covered_cells_vid)
                    break
            
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                best_episode_cells = info['covered_cells_gid']
            
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
                self.test(self.test_episodes, starting_loc, policy=self.policy)
                
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(self.env.city.agg_od_mx())
            
            if not self.env.city.ignore_existing_lines:
                for i, l in enumerate(self.env.city.existing_lines):
                    station_locs = self.env.city.vector_to_grid(l)
                    ax.plot(station_locs[:, 1], station_locs[:, 0], '-o', color='#A1A9FF', label='Existing lines' if i == 0 else None)
            
            best_episode_cells = np.array(best_episode_cells)
            ax.plot(best_episode_cells[:, 1], best_episode_cells[:, 0], '-or', label='Generated line')
            # If the test episodes are more than 1, we can plot the average line

            self.highlight_cells([(best_episode_cells[0][0].item(), best_episode_cells[0][1].item())], ax=ax, color='limegreen')
            fig.suptitle(f'Best Episode line \n reward: {best_episode_reward}')
            fig.legend(loc='lower center', ncol=2)
            wandb.log({"Best-Episode-Line": wandb.Image(fig)})
            plt.close(fig)
        
            wandb.finish()
        return self.Q, rewards, avg_rewards, epsilons, best_episode_reward, best_episode_cells, actual_starting_locs


    def test(self, test_episodes, starting_loc=None, policy=None):
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
            episode_step = 0
            while True:
                state_index = self.env.city.grid_to_vector(state['location'][None, :]).item()
                if policy is not None:
                    action = policy[episode_step]
                else:
                    action = np.argmax(self.Q[state_index, :] - 10000000 * (1-info['action_mask'].astype(np.int64)))
                    action = action.item()
                
                actions.append(action)
                new_state, reward, done, _, info = self.env.step(action)
                locations.append(new_state['location'].tolist())
                reward = reward.sum()
                episode_reward += reward
                episode_step += 1
                state = new_state
                if done:
                    break
            total_rewards += episode_reward
            generated_lines.append(locations)
            
        if self.log:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(self.env.city.agg_od_mx())
            
            if not self.env.city.ignore_existing_lines:
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
        
        ## TODO DELETE these diagnostics
        print(f'Average reward over {test_episodes} episodes: {total_rewards/test_episodes}')
        print(f'Actions of last episode: {actions}')
        print(f'vids of last episode: {self.env.unwrapped.city.grid_to_vector(np.array(locations)).tolist()}')
