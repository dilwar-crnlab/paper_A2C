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
    def forward(self, x, adj):
        if adj.dim() ==3:
            adj=adj[0]
        adj_hat = adj + torch.eye(adj.size(0), device=adj.device)
        D = torch.diag(torch.sum(adj_hat, dim=1))
        D_inv_sqrt = torch.diag(torch.pow(torch.diag(D) + 1e-8, -0.5))
        support = torch.mm(torch.mm(D_inv_sqrt, adj_hat), D_inv_sqrt)
        output = torch.mm(torch.mm(support, x), self.weight)
        return F.relu(output)

class GCNModule(nn.Module):
   def __init__(self, num_nodes):
       super(GCNModule, self).__init__()
       self.gcn = GCNLayer(feature_dim=1)
       self.memory = None
   def forward(self, x, adj, switch_on):
       input_features = x if x is not None else self.memory
       if switch_on:
           out = self.gcn(input_features, adj)
           self.memory = out
       return self.memory
   def reset_memory(self):
       self.memory = None

class SimpleRNNLayer(nn.Module):
    def __init__(self, k_paaths, input_size, hidden_size=256):
        super(SimpleRNNLayer, self).__init__()
        self.k_paths= k_paaths
        self.feature_dim = input_size
        self.rnn = nn.RNN(
           input_size=input_size,  
           hidden_size=hidden_size,
           batch_first=True,
           #num_layers=1
        )
        

    
    def forward(self, path_sequences):
       # path_sequences: [batch_size, k_paths, P, feature_dim]
       batch_size = 1
       
       # Reshape to [1, k_paths, P, num_edges]
       path_sequences = path_sequences.unsqueeze(0)
       # Reshape for RNN: [1*k_paths, P, num_edges]
       rnn_input = path_sequences.view(batch_size*self.k_paths, -1, self.feature_dim)
        
       output, _ = self.rnn(rnn_input)
       final_hidden = output[:, -1, :]  # [batch_size*k_paths, hidden_size]
        
       return final_hidden.view(batch_size, -1)  # [1, k_paths*hidden_size]

class DeepRMSAFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, P, num_original_nodes=14, num_edges=44, k_paths=5, num_bands=2, hidden_size=128):
        super().__init__(observation_space, features_dim=hidden_size)
        self.num_original_nodes = num_original_nodes
        self.num_edges = num_edges 
        self.k_paths = k_paths
        self.P=P
        self.num_bands = num_bands
        self.hidden_size = hidden_size

        self.gcn_modules = nn.ModuleList([GCNModule(num_edges) for _ in range(k_paths)])
        # RNN with 256 neurons
        self.rnn = SimpleRNNLayer(k_paths, input_size=num_edges, hidden_size=256)

        # Calculate input size for FC layer
        fc_input_size = (256 +                    # RNN output (k_paths * hidden_size/k_paths)
                        2*num_original_nodes +     # Source-dest (28)
                        k_paths +                  # Slots (5)
                        k_paths*6 +                # C band features (30)
                        k_paths*6)                 # L band features (30)
                    # Total: 349
        
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
        source_dest = obs[:, idx:idx + 2*self.num_original_nodes]  # [1, 28]
        idx += 2*self.num_original_nodes

        slots = obs[:, idx:idx + self.k_paths]
        slots = obs[:, idx:idx + self.k_paths]  # [1, 5]

        idx += self.k_paths


         # Extract and reshape spectrum features for C and L bands
        spectrum_raw = obs[:, idx:idx + self.k_paths*6*self.num_bands]
        # Reshape to [batch_size, k_paths, num_bands, 6_features]
        spectrum = spectrum_raw.view(batch_size, self.k_paths, self.num_bands, 6)
        
        # Separate C and L band features
        c_band_features = spectrum[:, :, 0, :]  # [1, 5, 6]
        l_band_features = spectrum[:, :, 1, :]  # [1, 5, 6]
        #print("C band ", c_band_features)
        #print("L band ", l_band_features)
        idx += self.k_paths*6*self.num_bands

        feature_matrix = obs[:, idx:idx + self.num_edges*self.k_paths].reshape(batch_size, self.k_paths, self.num_edges)
        idx += self.num_edges*self.k_paths

        adj_matrix = obs[:, idx:].reshape(batch_size, self.num_edges, self.num_edges)

        rnn_inputs= []
        path_sequences = []
        for i in range(self.k_paths):
            print("For path ------------------------------------------------------------------------------",i)
            path_features = feature_matrix[:, i, :]
            path_features= path_features.t()
            print("Path_feature shape", path_features.shape)
            path_length = torch.sum(path_features != 0).item()    
            path_outputs = []      
            for step in range(path_length):
                #print("For step----------------------------------------------------------------------", step)
                input_features = path_features if step == 0 else None
                #print("Input features", input_features.shape)
                switch_on = step < path_length
                #print("Switch on", switch_on)
                out = self.gcn_modules[i](input_features, adj_matrix, switch_on)
                path_outputs.append(out)

            # Continue propagating final memory value until P steps
            final_memory = self.gcn_modules[i].memory
            for _ in range(path_length, self.P):
                path_outputs.append(final_memory)   

            # Stack P outputs for this path
            path_sequence = torch.stack(path_outputs, dim=0)
            path_sequences.append(path_sequence)
            #print("path_sequence", path_sequence)
            #print("path_sequence shape", path_sequence.shape)
            
        
        # Stack all paths' sequences
        # Process through RNN
        rnn_inputs = torch.stack(path_sequences, dim=0)  # [k_paths, P, num_edges]
        rnn_out = self.rnn(rnn_inputs)  # [batch_size, k_paths*hidden_size]
        print("RNN inputs", rnn_inputs.shape)
        
        print("rnn_out shape:", rnn_out.shape)
        print("source_dest shape:", source_dest.shape)
        print("slots shape:", slots.shape)
        print("spectrum shape:", spectrum.shape)

        #combined = torch.cat([rnn_out, source_dest, slots, spectrum], dim=1)
        # Combine all features with separate C and L band features
        combined = torch.cat([
            rnn_out,                    # Path features from RNN
            source_dest,                # Source-destination encoding
            slots,                      # Slots per path
            c_band_features.reshape(batch_size, -1),        # [batch_size, k_paths*6]
            l_band_features.reshape(batch_size, -1)         # [batch_size, k_paths*6] 
        ], dim=1)
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

    k_shortest_paths = topology.graph["ksp"]
    
    def find_longest_path_by_hops(k_shortest_paths):
        longest_path = None
        max_hops = 0
        source_dest = None
        # Iterate through all source-destination pairs
        for (src, dst), paths in k_shortest_paths.items():
            # Check each path for this source-destination pair
            for path in paths:
                # Number of hops is the number of nodes minus 1
                num_hops = len(path.node_list) - 1
                if num_hops > max_hops:
                    max_hops = num_hops
                    longest_path = path
                    source_dest = (src, dst)
        return max_hops
    logest_path=find_longest_path_by_hops(k_shortest_paths)
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
            P=logest_path,
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
