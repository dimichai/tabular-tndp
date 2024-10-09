#%%
import json
import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
# Set the color palette to start with two specific colors using matplotlib

start_colors = ["#ff7f5e", "#1f77b4"]  # Blue and orange
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
custom_colors = start_colors + [c for c in default_colors if c not in start_colors]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)



PROJECT_NAME = 'TNDP-RL'
REQ_SEEDS = 5 # to control if a model was not run for sufficient seeds
api = wandb.Api()

def read_json(file_path):
    with open(file_path, 'r') as file:
        json_content = json.load(file)
    return json_content

def gini(x, normalized=True):
    sorted_x = np.sort(x, axis=1)
    n = x.shape[1]
    cum_x = np.cumsum(sorted_x, axis=1, dtype=float)
    gi = (n + 1 - 2 * np.sum(cum_x, axis=1) / cum_x[:, -1]) / n
    if normalized:
        gi = gi * (n / (n - 1))
    return gi

def average_per_step(hvs_by_seed):
    # Determine the maximum length of the sublists
    max_length = max(len(sublist) for sublist in hvs_by_seed)
    
    # Pad shorter sublists with zeros
    padded_hvs_by_seed = [sublist + [sublist[-1]] * (max_length - len(sublist)) for sublist in hvs_by_seed]
    
    # Calculate the average per step
    averages = []
    for i in range(max_length):
        step_values = [sublist[i] for sublist in padded_hvs_by_seed]
        averages.append(sum(step_values) / len(hvs_by_seed))
    
    return averages

def load_all_results_from_wadb(all_objectives, env_name=None):        
    all_results = pd.DataFrame()
    avg_reward_over_time = pd.DataFrame()

    for _, reward_type in enumerate(all_objectives):
        models_to_plot = pd.DataFrame(reward_type['models'])

        results_by_reward_type = {}
        for i, model_name in enumerate(models_to_plot['name'].unique()):
            models = models_to_plot[models_to_plot['name'] == model_name].to_dict('records')
            if len(models[0]['run_ids']) < REQ_SEEDS:
                print(f"!WARNING! {reward_type['reward_type']} reward type, {model_name} does not have enough seeds (has {len(models[0]['run_ids'])}, while {REQ_SEEDS} are required)")

            avg_rewards_by_seed = []
            for j, model in enumerate(models):
                print(f"Processing {env_name} {model_name} ({reward_type['reward_type']}) - {model['run_ids']}")
                # Read the content of the output file
                results_by_reward_type[model_name] = {'average_test_reward': []}
                for i in range(len(model['run_ids'])):
                    if model['run_ids'][i] == '':
                        print(f"WARNING - Empty run id in {model_name}")
                        continue
                    
                    run = api.run(f"{PROJECT_NAME}/{model['run_ids'][i]}")

                    results_by_reward_type[model_name]['average_test_reward'].append(run.summary['Average-Test-Reward'])

                    if model_name != 'DeepRL':
                        keys_to_load = ['episode', 'average_reward']
                        history = []
                        for row in run.scan_history(keys=keys_to_load):
                            history.append(row)

                        # Convert to DataFrame
                        history = pd.DataFrame(history)
                        rewards = history[history['average_reward'] > 0]['average_reward'].tolist()
                        
                        if len(rewards) > 0:
                            avg_rewards_by_seed.append(rewards)
                        else:
                            print(f"WARNING - No average_reward values in {model_name}, {reward_type} - {model['run_ids'][i]}")
                    else:
                        rewards = []
                        

            model_name_adj = model_name.replace(f'-{env_name}', '')
            if len(avg_rewards_by_seed) > 0:
                averages = average_per_step(avg_rewards_by_seed)
                avg_reward_over_time = pd.concat([avg_reward_over_time, pd.DataFrame({f"{model_name_adj}_{reward_type['reward_type']}": averages})])
            
        # Quite a hacky way to get the results in a dataframe, but didn't have time to do it properly (thanks copilot)
        # Convert all_results to a dataframe, with columns 'model', 'metric', 'value', and each row is a different value and not a list
        # results_by_objective = pd.DataFrame([(name, metric, value) for name in results_by_objective.keys() for metric in results_by_objective[name].keys() for value in results_by_objective[name][metric]], columns=['model', 'metric', 'value'])
        # Convert all_results to a dataframe, with columns 'model', 'lambda; 'metric', 'value', and each row is a different value and not a list
        results_by_reward_type = pd.DataFrame([(reward_type['reward_type'], name, metric, value) for name in results_by_reward_type.keys() 
                                            for metric in results_by_reward_type[name].keys() 
                                            for value in results_by_reward_type[name][metric]], columns=['reward_type', 'model', 'metric', 'value'])
        results_by_reward_type['model'] = results_by_reward_type['model'].str.replace(f'-{env_name}', '')
        all_results = pd.concat([all_results, results_by_reward_type])
        
    return all_results, avg_reward_over_time

xian_results, xian_avg_rw_over_time = load_all_results_from_wadb(read_json('./result_ids_xian.txt'), 'Xian')
# %%
# Calculate mean and confidence interval (CI)
def mean_ci(series):
    mean = series.mean()
    std_err = np.std(series, ddof=1) / np.sqrt(len(series))
    ci_max = mean + std_err * 1.96
    ci_min = mean - std_err * 1.96
    return pd.Series([mean, ci_min, ci_max], index=['mean', 'ci_min', 'ci_max'])

xian_results_summary = xian_results.groupby(['reward_type', 'model', 'metric'])['value'].apply(mean_ci).unstack().reset_index()
xian_results_summary

#%%
# Plot a bar chart of xian_results_summary with error bars
fig, axs = plt.subplots(2, 2, figsize=(9, 6))
axs = axs.flatten()

reward_types = ['max_efficiency', 'ggi2', 'ggi4', 'rawls']
reward_type_names = ['Max Efficiency', 'GGI(2)', 'GGI(4)', 'Rawls']

for idx, reward_type in enumerate(reward_types):
    # Filter data for the current reward type
    data = xian_results_summary[xian_results_summary['reward_type'] == reward_type]
        
    # Plot bars with error bars
    for i, (model, group) in enumerate(data.groupby('model')):
        axs[idx].bar(i, group['mean'], 
                     yerr=[group['mean'] - group['ci_min'], group['ci_max'] - group['mean']], 
                     capsize=5, label=model)
    
    # Set labels and title for each subplot
    axs[idx].set_xticks(range(len(data['model'].unique())))
    axs[idx].set_xticklabels(data['model'].unique())
    axs[idx].set_ylabel('Value')
    axs[idx].set_title(f'{reward_type_names[idx]}')
    
plt.tight_layout()
plt.show()

#%% Plot average reward over time
wandb_to_local_mapper = [
    {
        "reward_type": "max_efficiency",
        "mapping": {
            'eng3ru2n': 'xian_20241003_11_05_54.565620',
            'fgor0rft': 'xian_20241003_11_06_57.001302',
            'hxqtppx7': 'xian_20241002_21_32_09.662719',
            'hv9p3c2u': 'xian_20241002_21_31_53.751992',
            'j0ejk07m': 'xian_20241003_11_10_45.833755',
        }
    }
]

result_dir = '../../fair-network-expansion/result/new'

fig, ax = plt.subplots(figsize=(8, 5))

reward_type = 'max_efficiency'
reward_type_name = 'Max Efficiency'

# Add lines from the mapper first
mapper = next((m for m in wandb_to_local_mapper if m['reward_type'] == reward_type), None)
if mapper:
    mapping = mapper['mapping']
    all_reward_data = []
    for wandb_id, local_dir in mapping.items():
        file_path = f"{result_dir}/{local_dir}/reward_actloss_criloss.txt"
        try:
            reward_data = np.loadtxt(file_path)
            all_reward_data.append(reward_data[:, 0])
        except FileNotFoundError:
            print(f"File not found: {file_path}")
    
    if all_reward_data:
        avg_reward_data = np.mean(all_reward_data, axis=0)
        ax.semilogx(np.arange(1, len(avg_reward_data) + 1), avg_reward_data, label="DeepRL", linewidth=2)

# Then plot the other data
data = xian_avg_rw_over_time.filter(regex=f'.*{reward_type}')

for column in data.columns:
    interval_data = data[column].iloc[::128].reset_index(drop=True)
    ax.semilogx(interval_data.index + 1, interval_data, label='TabularMNEP', linewidth=2)

# Set labels and title
ax.set_xlabel('Steps (x128 episodes)')
ax.set_ylabel('Average Reward')
ax.set_title(f'Average Reward Over Time - {reward_type_name}')

# Adjust legend
ax.legend(loc='lower right', fontsize=16)

plt.tight_layout()
plt.show()

# %%
