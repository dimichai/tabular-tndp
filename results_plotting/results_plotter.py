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
    
    # Calculate the average per step and confidence interval
    averages = []
    mins = []
    maxs = []
    for i in range(max_length):
        step_values = [sublist[i] for sublist in padded_hvs_by_seed]
        average = sum(step_values) / len(hvs_by_seed)
        averages.append(average)
        # Calculate the confidence interval
        ci = 1.96 * np.std(step_values) / np.sqrt(len(step_values))
        mins.append(average - ci)
        maxs.append(average + ci)
    
    return averages, mins, maxs

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
                averages, ci_upper, ci_lower = average_per_step(avg_rewards_by_seed)
                
                avg_reward_over_time = pd.concat([avg_reward_over_time, 
                                                  pd.DataFrame({f"{model_name_adj}_{reward_type['reward_type']}": averages,
                                                                f"{model_name_adj}_{reward_type['reward_type']}_upper": ci_upper,
                                                                f"{model_name_adj}_{reward_type['reward_type']}_lower": ci_lower})])
            
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
ams_results, ams_avg_rw_over_time = load_all_results_from_wadb(read_json('./result_ids_ams.txt'), 'Amsterdam')
# %%
# Calculate mean and confidence interval (CI)
def mean_ci(series):
    mean = series.mean()
    std_err = np.std(series, ddof=1) / np.sqrt(len(series))
    ci_max = mean + std_err * 1.96
    ci_min = mean - std_err * 1.96
    return pd.Series([mean, ci_min, ci_max], index=['mean', 'ci_min', 'ci_max'])

xian_results_summary = xian_results.groupby(['reward_type', 'model', 'metric'])['value'].apply(mean_ci).unstack().reset_index()

ams_results_summary = ams_results.groupby(['reward_type', 'model', 'metric'])['value'].apply(mean_ci).unstack().reset_index()

#%%
# Plot a bar chart of xian_results_summary with error bars
def plot_results_summary(results_summary, city_name):
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    axs = axs.flatten()

    reward_types = ['max_efficiency', 'ggi2', 'ggi4', 'rawls']
    reward_type_names = ['Max Efficiency', 'GGI(2)', 'GGI(4)', 'Rawls']

    for idx, reward_type in enumerate(reward_types):
        # Filter data for the current reward type
        data = results_summary[results_summary['reward_type'] == reward_type]
            
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
    
    # plt.suptitle(f'Results {city_name}')
    plt.tight_layout()
    plt.show()

# Usage:
plot_results_summary(xian_results_summary, 'Xian')
plot_results_summary(ams_results_summary, 'Amsterdam')

#%% Plot average reward over time
wandb_to_local_mapper = [
    {
        "environment": "xian",
        "reward_type": "max_efficiency",
        "mapping": {
            'eng3ru2n': 'xian_20241003_11_05_54.565620',
            'fgor0rft': 'xian_20241003_11_06_57.001302',
            'hxqtppx7': 'xian_20241002_21_32_09.662719',
            'hv9p3c2u': 'xian_20241002_21_31_53.751992',
            'j0ejk07m': 'xian_20241003_11_10_45.833755',
        }
    },
    {
        "environment": "xian",
        "reward_type": "rawls",
        "mapping": {
            'eng3ru2n': 'xian_20241004_13_04_11.931713',
            'fgor0rft': 'xian_20241004_13_04_21.155581',
            'hxqtppx7': 'xian_20241004_16_34_34.483372',
            'hv9p3c2u': 'xian_20241004_16_34_34.482076',
            'j0ejk07m': 'xian_20241004_16_34_47.456490',
        }
    },
    {
        "environment": "amsterdam",
        "reward_type": "max_efficiency",
        "mapping": {
            '1se01ud5': 'amsterdam_20241010_12_22_01.564520',
            '5sif9a7w': 'amsterdam_20241010_12_22_01.568074',
            'z4a9e3fw': 'amsterdam_20241010_12_22_01.570698',
            '6ot0zo6e': 'amsterdam_20241010_12_22_01.567504',
            'odcongwb': 'amsterdam_20241010_14_40_41.815377',
        }
    },
    {
        "environment": "amsterdam",
        "reward_type": "rawls",
        "mapping": {
            'xz7id1ah': 'amsterdam_20241010_14_42_47.909724',
            '9crampld': 'amsterdam_20241011_08_00_43.281005',
            'qwzu05ht': 'amsterdam_20241011_08_32_46.147697',
            '425us7iw': 'amsterdam_20241011_09_05_22.118755',
            'fxnv8t7d': 'amsterdam_20241011_09_31_59.329177',
        }
    },
]

result_dir = '../../fair-network-expansion/result/new'

fig, axs = plt.subplots(1, 2, figsize=(16, 5))

reward_type = 'max_efficiency'
reward_type_name = 'Max Efficiency'

# Plotting for Xian
ax = axs[0]
mapper = next((m for m in wandb_to_local_mapper if m['reward_type'] == reward_type and m['environment'] == 'xian'), None)
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
        ci = 1.96 * np.std(all_reward_data, axis=0) / np.sqrt(len(all_reward_data))
        ax.semilogx(np.arange(1, len(avg_reward_data) + 1), avg_reward_data, label="DeepRL", linewidth=2)
        ax.fill_between(np.arange(1, len(avg_reward_data) + 1), avg_reward_data - ci, avg_reward_data + ci, alpha=0.3)

# Then plot the other data
data = xian_avg_rw_over_time.filter(regex=f'.*{reward_type}').iloc[::128].reset_index(drop=True)
ax.semilogx(data.index + 1, data[f'TabularMNEP_{reward_type}'], label='TabularMNEP', linewidth=2)
ax.fill_between(data.index + 1, data[f'TabularMNEP_{reward_type}_lower'], data[f'TabularMNEP_{reward_type}_upper'], alpha=0.3)

# Plotting for Amsterdam
ax = axs[1]
mapper = next((m for m in wandb_to_local_mapper if m['reward_type'] == reward_type and m['environment'] == 'amsterdam'), None)
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
        ci = 1.96 * np.std(all_reward_data, axis=0) / np.sqrt(len(all_reward_data))
        ax.semilogx(np.arange(1, len(avg_reward_data) + 1), avg_reward_data, label="DeepRL", linewidth=2)
        ax.fill_between(np.arange(1, len(avg_reward_data) + 1), avg_reward_data - ci, avg_reward_data + ci, alpha=0.3)

data = ams_avg_rw_over_time.filter(regex=f'.*{reward_type}').iloc[::128].reset_index(drop=True)
ax.semilogx(data.index + 1, data[f'TabularMNEP_{reward_type}'], label='TabularMNEP', linewidth=2)
ax.fill_between(data.index + 1, data[f'TabularMNEP_{reward_type}_lower'], data[f'TabularMNEP_{reward_type}_upper'], alpha=0.3)

# Set labels and title for both subplots
for ax in axs:
    ax.set_xlabel('Steps (x128 episodes)')
    ax.set_ylabel('Average Reward')
    ax.set_title(f'Average Reward Over Time - {reward_type_name}')
    ax.legend(loc='lower right', fontsize=16)

plt.tight_layout()
plt.show()

# %%
