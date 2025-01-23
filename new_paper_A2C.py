import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3.common.results_plotter import load_results, ts2xy


class TensorPrinter:
    """
    A class that contains methods to print tensors in a readable format.
    """
    @staticmethod
    def print_tensor_with_indices(tensor):
        """
        Method to print a PyTorch tensor with row and column indices.
        """
        num_rows, num_cols = tensor.shape
        print("Tensor Matrix:")
        # Print column headers
        header = " " + " ".join(f"{col:2}" for col in range(num_cols))
        print(header)
        print("-" * len(header))
        
        # Print rows with indices
        for row_idx in range(num_rows):
            row_values = " ".join(f"{value:3.1f}" for value in tensor[row_idx])
            print(f"Row {row_idx:2}: {row_values}")

class SaveOnBestTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int, log_dir: str, verbose=1, show_plot: bool=False):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.show_plot = show_plot

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:

        if self.show_plot and self.n_calls % self.check_freq == 0 and self.n_calls > 5001:
            plotting_average_window = 100
            training_data = pd.read_csv(self.log_dir + 'training.monitor.csv', skiprows=1)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 4.8))
            ax1.plot(np.convolve(training_data['r'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax2.semilogy(np.convolve(training_data['episode_service_blocking_rate'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Episode service blocking rate')
            ax3.semilogy(np.convolve(training_data['episode_bit_rate_blocking_rate'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Episode bit rate blocking rate')
            # fig.get_size_inches()
            plt.tight_layout()
            plt.show()
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                 # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {} - ".format(self.num_timesteps), end="")
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                  # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)
                if self.verbose > 0:
                    clear_output(wait=True)

        return True


class GCNLayer(nn.Module):

    def __init__(self, feature_dim=1):
       super(GCNLayer, self).__init__()
       self.weight = nn.Parameter(torch.FloatTensor(feature_dim, feature_dim))
       nn.init.xavier_uniform_(self.weight)
       #print("Weight ,matrix", self.weight)
   
    def forward(self, x, adj):
        x = x.t()
        print("X", x.shape)
        TensorPrinter.print_tensor_with_indices(x)
        #print("adj matrix in GCNLayer forward", adj.shape)

        if adj.dim() ==3:
            adj=adj[0]
        #print("adj matrix in GCNLayer forward", adj.shape)

        adj_hat = adj + torch.eye(adj.size(0), device=adj.device)
        #print(adj_hat.shape)
        D = torch.diag(torch.sum(adj_hat, dim=1))
        D_inv_sqrt = torch.diag(torch.pow(torch.diag(D) + 1e-8, -0.5))
        #print(D)
        support = torch.mm(torch.mm(D_inv_sqrt, adj_hat), D_inv_sqrt)
        #TensorPrinter.print_tensor_with_indices(support)
        output = torch.mm(torch.mm(support, x), self.weight)
        return F.relu(output)

class GCNModule(nn.Module):
   def __init__(self, num_nodes):
       super(GCNModule, self).__init__()
       self.gcn = GCNLayer(feature_dim=1)
       self.memory = None
       
   def forward(self, x, adj, switch_on):
       input_features = x if x is not None else self.memory
       print("Input in GCN Module forward", input_features )
       if switch_on:
           out = self.gcn(input_features, adj)
           self.memory = out
       return self.memory
   
   def reset_memory(self):
       self.memory = None

class SimpleRNNLayer(nn.Module):
   def __init__(self, input_size, hidden_size):
       super(SimpleRNNLayer, self).__init__()
       self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
       
   def forward(self, x):
       output, _ = self.rnn(x)
       return output[:, -1, :]

class DeepRMSAFeatureExtractor(BaseFeaturesExtractor):
   def __init__(self, observation_space, num_original_nodes=14, num_edges=44, k_paths=5, num_bands=2, hidden_size=128):
       super().__init__(observation_space, features_dim=hidden_size)
       self.num_original_nodes = num_original_nodes
       self.num_edges = num_edges 
       self.k_paths = k_paths
       self.num_bands = num_bands
       self.hidden_size = hidden_size
       
       self.gcn_modules = nn.ModuleList([GCNModule(num_edges) for _ in range(k_paths)])
       
       self.rnn = SimpleRNNLayer(hidden_size, hidden_size)
       
       fc_input_size = (hidden_size + 
                       2*num_original_nodes +
                       k_paths + 
                       k_paths*6*num_bands)
       
       self.fc = nn.Sequential(
           nn.Linear(fc_input_size, hidden_size),
           nn.ReLU(),
           nn.Linear(hidden_size, hidden_size),
           nn.ReLU(),
           nn.Linear(hidden_size, hidden_size),
           nn.ReLU(),
           nn.Linear(hidden_size, hidden_size),
           nn.ReLU(),
           nn.Linear(hidden_size, hidden_size),
           nn.ReLU()
       )
       
   def forward(self, obs):
       batch_size = obs.shape[0]
       idx = 0
       source_dest = obs[:, idx:idx + 2*self.num_original_nodes]
       idx += 2*self.num_original_nodes
       
       slots = obs[:, idx:idx + self.k_paths]
       idx += self.k_paths
       
       spectrum = obs[:, idx:idx + self.k_paths*6*self.num_bands]
       idx += self.k_paths*6*self.num_bands
       
       feature_matrix = obs[:, idx:idx + self.num_edges*self.k_paths].reshape(batch_size, self.k_paths, self.num_edges)
       idx += self.num_edges*self.k_paths
       
       adj_matrix = obs[:, idx:].reshape(batch_size, self.num_edges, self.num_edges)
       #print("Adj matrix shape",adj_matrix.shape)
       
       gcn_outputs = []
       for i in range(self.k_paths):
           print("--------------------")
           print("Path ", i)
           path_features = feature_matrix[:, i, :]
           path_length = torch.sum(path_features != 0).item()
           #print("path length", path_length)
           
           for step in range(path_length):
               print("For step----------------------------------------------------------------------", step)
               input_features = path_features if step == 0 else None
               #print("Input features", input_features.shape)
               switch_on = step < path_length
               #print("Switch on", switch_on)
               out = self.gcn_modules[i](input_features, adj_matrix, switch_on)
               
           gcn_outputs.append(out)
       
       gcn_outputs = torch.stack(gcn_outputs, dim=1)
       rnn_out = self.rnn(gcn_outputs)
       combined = torch.cat([rnn_out, source_dest, slots, spectrum], dim=1)
       return self.fc(combined)

class DeepRMSAPolicy(ActorCriticPolicy):
   def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
       super(DeepRMSAPolicy, self).__init__(
           observation_space,
           action_space,
           lr_schedule,
           **kwargs
       )
       self.policy_net = nn.Linear(self.features_dim, action_space.n)
       self.value_net = nn.Linear(self.features_dim, 1)
       
   def forward(self, obs, deterministic=False):
       features = self.extract_features(obs)
       action_logits = self.policy_net(features)
       action_probs = F.softmax(action_logits, dim=1)
       value = self.value_net(features)
       return action_probs, value




def main():
   topology_name = 'nsfnet_chen'
   k_paths = 5
   with open(f'../topologies/{topology_name}_{k_paths}-paths_new.h5', 'rb') as f:
       topology = pickle.load(f)

   env_args = dict(
       num_bands=2,
       topology=topology, 
       seed=10,
       allow_rejection=False,
       j=1,
       mean_service_holding_time=25.0,
       mean_service_inter_arrival_time=0.1,
       k_paths=k_paths,
       episode_length=100,
       node_request_probabilities=None
   )

   log_dir = "./tmp/deeprmsa-a2c/"
   os.makedirs(log_dir, exist_ok=True)
   callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir, show_plot=False)
   
   monitor_info_keywords = (
       "service_blocking_rate",
       "episode_service_blocking_rate",
       "bit_rate_blocking_rate",
       "episode_bit_rate_blocking_rate",
   )

   env = gym.make('DeepRMSA-v0', **env_args)
   env = Monitor(env, log_dir + 'training', info_keywords=monitor_info_keywords)

   policy_kwargs = dict(
       features_extractor_class=DeepRMSAFeatureExtractor,
       features_extractor_kwargs=dict(
           num_original_nodes=topology.number_of_nodes(),
           num_edges=topology.number_of_edges(),
           k_paths=k_paths,
           num_bands=env_args['num_bands'],
           hidden_size=128
       ),
   )

   model = A2C(
       policy=DeepRMSAPolicy,
       env=env,
       learning_rate=1e-4,
       n_steps=5,
       gamma=0.95,
       ent_coef=0.01,
       vf_coef=0.5,
       max_grad_norm=0.5,
       tensorboard_log="./tb/A2C-DeepRMSA-v0/",
       policy_kwargs=policy_kwargs,
       verbose=1
   )

   model.learn(
       total_timesteps=1000000,
       callback=callback
   )

   model.save(f"{log_dir}/final_model")

if __name__ == "__main__":
   main()
