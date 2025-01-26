import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3.common.results_plotter import load_results, ts2xy

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

# ------------------- GCN Layer -------------------
class GCNLayer(nn.Module):
    """
    Simple GCN layer that expects:
      x:   [num_nodes, feature_dim]
      adj: [num_nodes, num_nodes]
    """
    def __init__(self, feature_dim=1):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(feature_dim, feature_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # If adjacency has shape [batch_size, num_nodes, num_nodes], just take the first
        if adj.dim() == 3:
            adj = adj[0]

        adj_hat = adj + torch.eye(adj.size(0), device=adj.device)
        D = torch.diag(torch.sum(adj_hat, dim=1))
        D_inv_sqrt = torch.diag(torch.pow(torch.diag(D) + 1e-8, -0.5))
        support = torch.mm(torch.mm(D_inv_sqrt, adj_hat), D_inv_sqrt)
        output = torch.mm(torch.mm(support, x), self.weight)
        return F.relu(output)


# ------------------- Recurrent GCN Module (per path) -------------------
class GCNModule(nn.Module):
    """
    Wraps one GCN layer with a "memory" so we can repeatedly apply it
    for P "circulations" (time steps). If switch_on=False, we re-use
    the last memory (not updating).
    """
    def __init__(self, num_nodes):
        super(GCNModule, self).__init__()
        self.gcn = GCNLayer(feature_dim=1)
        self.memory = None

    def forward(self, x, adj, switch_on):
        """
        x: [num_nodes, 1] or None.
        adj: [num_nodes, num_nodes].
        switch_on: bool indicating whether to update memory.
        """
        input_features = x if x is not None else self.memory
        if switch_on:
            out = self.gcn(input_features, adj)
            self.memory = out
        return self.memory  # Always return current memory

    def reset_memory(self):
        self.memory = None


# ------------------- Simple RNN Aggregation -------------------
class SimpleRNNLayer(nn.Module):
    """
    Expects input of shape [batch_size, k_paths, P, feature_dim_for_each_step]
    and returns [batch_size, k_paths * hidden_size].
    """
    def __init__(self, k_paaths, input_size, hidden_size=256):
        super(SimpleRNNLayer, self).__init__()
        self.k_paths = k_paaths
        self.feature_dim = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(
           input_size=input_size,  
           hidden_size=hidden_size,
           batch_first=True
        )

    def forward(self, path_sequences):
        """
        path_sequences shape: [batch_size, k_paths, P, feature_dim]

        We'll reshape to [batch_size*k_paths, P, feature_dim], run RNN,
        then reshape final hidden state to [batch_size, k_paths*hidden_size].
        """
        batch_size = path_sequences.size(0)
        # Flatten the (batch, k_paths) dimension => batch_size*k_paths
        rnn_input = path_sequences.view(
            batch_size * self.k_paths,
            path_sequences.size(2),   # P
            path_sequences.size(3)    # feature_dim
        )
        output, _ = self.rnn(rnn_input)
        # Grab the last time-step’s output => [batch_size*k_paths, hidden_size]
        final_hidden = output[:, -1, :]

        # Reshape to [batch_size, k_paths*hidden_size]
        return final_hidden.view(batch_size, -1)


# ------------------- SB3 Features Extractor -------------------
class DeepRMSAFeatureExtractor(BaseFeaturesExtractor):
    """
    1) Parse an observation that contains:
       - source_dest, slots, spectrum C/L band info
       - feature_matrix (k_paths x num_edges)
       - adjacency matrix
    2) For each path, unroll GCN up to P steps, collecting outputs
    3) Pass all path sequences to RNN => aggregated vector
    4) Concatenate with other scalar features => final FC
    """
    def __init__(
        self,
        observation_space,
        P,
        num_original_nodes=14,
        num_edges=44,
        k_paths=5,
        num_bands=2,
        hidden_size=128
    ):
        super().__init__(observation_space, features_dim=hidden_size)

        self.num_original_nodes = num_original_nodes
        self.num_edges = num_edges
        self.k_paths = k_paths
        self.P = P
        self.num_bands = num_bands
        self.hidden_size = hidden_size
        print("Hidden", self.hidden_size)

        # K GCN modules
        self.gcn_modules = nn.ModuleList([GCNModule(num_edges) for _ in range(k_paths)])
        # RNN aggregator
        self.rnn = SimpleRNNLayer(
            k_paaths=k_paths,
            input_size=num_edges,  # We'll flatten 44x1 => 44
            hidden_size=256
        )

        # Compute size for FC input
        # = RNN output (k_paths * 256) + 2*num_original_nodes + k_paths + k_paths*6 + k_paths*6
        # = (k_paths*256) + 28 + 5 + 30 + 30 = 1283 if k_paths=5
        fc_input_size = (
            (k_paths * 256) +
            (2 * num_original_nodes) +
             k_paths +
            (k_paths * 6) +
            (k_paths * 6)
        )

        # A multi-layer FC
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
        #print("batch size", batch_size)
        indx=0
        #print("OBS", obs.shape)
        #print("First 28 elements of obs tensor:", obs[0, :28])
        # Print the first 14 elements
        #print("First 14 elements of obs tensor (Source One-Hot):", obs[0, indx:indx + 14])
        indx += 14

        # Print the next 14 elements
        #print("Next 14 elements of obs tensor (Destination One-Hot):", obs[0, indx:indx + 14])
        indx += 14

        # Print the next 5 elements
        #print("Next 5 elements of obs tensor (Slots):", obs[0, indx:indx + 5])
        indx += 5

        # Print the next 60 elements
        #print("Next 60 elements of obs tensor (Spectrum):", obs[0, indx:indx + 60])
        indx += 60
        #print("indx", indx)

        # Print the next 60 elements
        #print("Next 220 elements of obs tensor (Spectrum):", obs[0, indx:indx + 220])
        indx += 220
        

        # feature_matrix_flattened = obs[:, 93:93 + self.num_edges * self.k_paths]
        # feature_matrix = feature_matrix_flattened.view(batch_size, self.num_edges, self.k_paths)
        # #print("Reconstructed Feature Matrix (44x5):", feature_matrix)

        # 1) Parse source-dest
        source_dest = obs[:, idx : idx + 2*self.num_original_nodes]
        #print("Src Dst", source_dest)
        reshaped_tensor = source_dest.view(-1, 14)  # Reshape to (1, 14) if needed
        original_numbers = [torch.argmax(reshaped_tensor[i]).item()+1 for i in range(reshaped_tensor.size(0))]
        #print("Src Dst", original_numbers)

        idx += 2*self.num_original_nodes

        # 2) Parse slots
        slots = obs[:, idx : idx + self.k_paths]  # shape [batch_size, k_paths]
        idx += self.k_paths
        #print("slots", slots)

        # 3) Parse spectrum for C & L bands
        spectrum_raw = obs[:, idx : idx + self.k_paths*6*self.num_bands]
        idx += self.k_paths*6*self.num_bands
        spectrum = spectrum_raw.view(batch_size, self.k_paths, self.num_bands, 6)
        #print("Spectrum in forward", spectrum_raw)

        # separate C/L band
        c_band_features = spectrum[:, :, 0, :]  # shape [batch_size, k_paths, 6]
        l_band_features = spectrum[:, :, 1, :]  # shape [batch_size, k_paths, 6]

        # 4) Feature matrix: [batch_size, k_paths, num_edges]
        feature_matrix = obs[:, idx : idx + self.num_edges * self.k_paths]
        #print("FM(X)", feature_matrix)
        feature_matrix = feature_matrix.view(batch_size, self.num_edges, self.k_paths)
        idx += self.num_edges * self.k_paths

        # 5) Adjacency: [batch_size, num_edges, num_edges]
        adj_matrix = obs[:, idx:].view(batch_size, self.num_edges, self.num_edges)

        # 6) Build GCN “unrolled” sequences per path
        #    We want shape => [batch_size, k_paths, P, num_edges, 1]
        all_path_sequences = []
        for b in range(batch_size):
            # Reset GCN memory for each new sample in batch
            for module in self.gcn_modules:
                module.reset_memory()

            # Collect the P-step sequences for each path
            path_outputs_for_batch = []
            for i in range(self.k_paths):
                # path_features: [num_edges] => reshape [num_edges, 1]
                #print("For path----------------------------------------------------------",i)
                pf = feature_matrix[b, :, i].unsqueeze(-1)
                #print("X",pf.t())
                # Count how many edges are nonzero (or define your path_length differently)
                path_length = torch.sum(pf != 0).item()
                #print("path length", path_length)

                single_path_outputs = []
                for step in range(self.P):
                    if step == 0:
                        # first step uses pf
                        out = self.gcn_modules[i](pf, adj_matrix[b], switch_on=(step < path_length))
                        #print("Out in step=0", out.t())
                    else:
                        # subsequent steps pass None
                        out = self.gcn_modules[i](None, adj_matrix[b], switch_on=(step < path_length))

                    single_path_outputs.append(out)  # each out: [num_edges, 1]

                # stack => [P, num_edges, 1]
                single_path_seq = torch.stack(single_path_outputs, dim=0)
                path_outputs_for_batch.append(single_path_seq)

            # shape => [k_paths, P, num_edges, 1]
            path_outputs_for_batch = torch.stack(path_outputs_for_batch, dim=0)
            # Collect for this batch element
            all_path_sequences.append(path_outputs_for_batch)

        # Now stack across batch => [batch_size, k_paths, P, num_edges, 1]
        all_path_sequences = torch.stack(all_path_sequences, dim=0)

        # 7) Pass to RNN aggregator
        #    RNN expects shape [batch_size, k_paths, P, feature_dim], so we flatten num_edges*1 =>  num_edges
        #    -> final shape for each time step = 44
        all_path_sequences = all_path_sequences.view(
            batch_size, self.k_paths, self.P, self.num_edges
        )
        rnn_out = self.rnn(all_path_sequences)  # => [batch_size, k_paths * hidden_size=256]
        #print("RNN out", rnn_out)

        # 8) Concatenate with the other features
        combined = torch.cat([
            rnn_out,  # [batch_size, k_paths*256]
            source_dest,                   # [batch_size, 2*num_original_nodes]
            slots,                         # [batch_size, k_paths]
            c_band_features.reshape(batch_size, -1),  # [batch_size, k_paths*6]
            l_band_features.reshape(batch_size, -1)   # [batch_size, k_paths*6]
        ], dim=1)

        # 9) Pass through the final MLP
        fc_out = self.fc(combined)
        print("FC out",fc_out.shape)
        return fc_out


# ------------------- SB3 Policy -------------------
class DeepRMSAPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(DeepRMSAPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )
        print("Action in Policy", action_space)
        self.policy_net = nn.Linear(self.features_dim, action_space.n)
        self.value_net = nn.Linear(self.features_dim, 1)

        print("Feature Extractor Output Size (features_dim):", self.features_dim)
        print("Policy Network Input Size (policy_net):", self.policy_net.in_features)
        print("Value Network Input Size (value_net):", self.value_net.in_features)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)

        # 2) Actor head -> unnormalized logits
        logits = self.policy_net(features)  # shape: [batch_size, n_actions]

        # 3) Critic -> value estimates
        values = self.value_net(features).flatten()  # shape: [batch_size]

        # 4) Build the categorical distribution from logits
        dist = Categorical(logits=logits)
        

        # 5) Decide how to pick actions:
        if deterministic:
            # Argmax if in 'deterministic' mode
            actions = torch.argmax(logits, dim=1)
        else:
            # Sample from the distribution
            actions = dist.sample()

        # 6) log_probs for the chosen actions
        log_probs = dist.log_prob(actions)

        # Inside the forward method of the policy
        print("Feature Extractor Output Shape:", features.shape)
        print("Logits Shape (Policy Output):", logits.shape)
        print("Values Shape (Critic Output):", values.shape)

        return actions, values, log_probs


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
