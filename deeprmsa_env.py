from typing import Tuple

import gym
import numpy as np

from .rmsa_env import RMSAEnv

from .optical_network_env import OpticalNetworkEnv

class DeepRMSAEnv(RMSAEnv):
    def __init__(
        self,
        num_bands,
        topology=None,
        j=1,
        episode_length=1000,
        mean_service_holding_time=25.0,
        mean_service_inter_arrival_time=0.1,
        node_request_probabilities=None,
        seed=None,
        k_paths=5,
        allow_rejection=False,
    ):
        super().__init__(
            num_bands=num_bands,
            topology=topology,
            episode_length=episode_length,
            load=mean_service_holding_time / mean_service_inter_arrival_time,
            mean_service_holding_time=mean_service_holding_time,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            k_paths=k_paths,
            allow_rejection=allow_rejection,
            reset=False,
        )

        self.j = j
        #shape = 1 + 2 * self.topology.number_of_nodes() + (2 * self.j + 3) * self.k_paths * self.num_bands

      
        shape = (
            2 * self.topology.number_of_nodes() +     # source_dest (2 one-hot vectors)
            self.k_paths +                            # slots_per_path
            (self.k_paths * 6 * self.num_bands) +     # spectrum distribution (6 features per band per path)
            (self.topology.number_of_edges() * self.k_paths) +  # feature matrix
            (self.topology.number_of_edges() * self.topology.number_of_edges())  # adjacency matrix
        )
        print("shape", shape)
    
        self.observation_space = gym.spaces.Box(
            low=0, high=1, 
            shape=(shape,), 
            dtype=np.uint8
        )

        #self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.uint8, shape=(shape,))
        self.action_space = gym.spaces.Discrete(self.k_paths  * self.num_bands * self.j + self.reject_action)
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.reset(only_episode_counters=False)

    def step(self, action: int):
        parent_step_result = None
        valid_action = False

        if action < self.k_paths * self.j * self.num_bands:  # action is for assigning a route
            valid_action = True
            route, band, block = self._get_route_block_id(action)

            initial_indices, lengths = self.get_available_blocks(route, self.num_bands, band, self.modulations)
            slots = self.get_number_slots(self.k_shortest_paths[self.current_service.source, self.current_service.destination][route], self.num_bands, band, self.modulations)
            if block < len(initial_indices):
                parent_step_result = super().step(
                    [route, band, initial_indices[block]])
            else:
                parent_step_result = super().step(
                    [self.k_paths, self.num_bands, self.num_spectrum_resources])
        else:
            parent_step_result = super().step(
                [self.k_paths, self.num_bands, self.num_spectrum_resources])

        obs, rw, _, info = parent_step_result
        info['slots'] = slots if valid_action else -1
        return parent_step_result

    # def observation(self):
    #     # observation space defined as in https://github.com/xiaoliangchenUCD/DeepRMSCA/blob/eb2f2442acc25574e9efb4104ea245e9e05d9821/DeepRMSCA_Agent.py#L384
    #     source_destination_tau = np.zeros((2, self.topology.number_of_nodes()))
    #     min_node = min(self.current_service.source_id, self.current_service.destination_id)
    #     max_node = max(self.current_service.source_id, self.current_service.destination_id)
    #     source_destination_tau[0, min_node] = 1
    #     source_destination_tau[1, max_node] = 1
    #     spectrum_obs = np.full((self.k_paths * self.num_bands, 2 * self.j + 3), fill_value=-1.)
    #     # for the k-path ranges all possible bands to take the best decision
    #     for idp, path in enumerate(self.k_shortest_paths[self.current_service.source, self.current_service.destination]):
    #       for band in range(self.num_bands):
    #         available_slots = self.get_available_slots(path, band)
    #         num_slots = self.get_number_slots(path, self.num_bands, band, self.modulations)
    #         initial_indices, lengths = self.get_available_blocks(idp, self.num_bands, band, self.modulations)
    #         for idb, (initial_index, length) in enumerate(zip(initial_indices, lengths)):
    #                     # initial slot index
    #             spectrum_obs[idp + (self.k_paths * band), idb * 2 + 0] = 2 * (initial_index - .5 * self.num_spectrum_resources) / self.num_spectrum_resources

    #                     # number of contiguous FS available
    #             spectrum_obs[idp + (self.k_paths * band), idb * 2 + 1] = (length - 8) / 8
    #         spectrum_obs[idp + (self.k_paths * band), self.j * 2] = (num_slots - 5.5) / 3.5 # number of FSs necessary

    #         idx, values, lengths = DeepRMSAEnv.rle(available_slots)

    #         av_indices = np.argwhere(values == 1) # getting indices which have value 1
    #         # spectrum_obs = matrix with shape k_routes x s_bands in the scenario
    #         spectrum_obs[idp + (self.k_paths * band), self.j * 2 + 1] = 2 * (np.sum(available_slots) - .5 * self.num_spectrum_resources) / self.num_spectrum_resources # total number of available FSs
    #         spectrum_obs[idp + (self.k_paths * band), self.j * 2 + 2] = (np.mean(lengths[av_indices]) - 4) / 4 # avg. number of FS blocks available
    #     bit_rate_obs = np.zeros((1, 1))
    #     bit_rate_obs[0, 0] = self.current_service.bit_rate / 100

    #     return np.concatenate((bit_rate_obs, source_destination_tau.reshape((1, np.prod(source_destination_tau.shape))),
    #                            spectrum_obs.reshape((1, np.prod(spectrum_obs.shape)))), axis=1)\
    #         .reshape(self.observation_space.shape)


    # def obs(self):
    #     src = self.current_service.source
    #     dst = self.current_service.destination
    #     # 1. Source-destination one-hot vectors
    #     source_one_hot = np.zeros(self.topology.number_of_nodes())
    #     dest_one_hot = np.zeros(self.topology.number_of_nodes())
        
    #     # Use node indices for one-hot encoding
    #     source_one_hot[self.topology.nodes[src]['index']] = 1
    #     dest_one_hot[self.topology.nodes[dst]['index']] = 1

    #     feature_matrix = self.global_TSG[src, dst]['feature_matrix']  # X in the paper
    #     adjacency_matrix=self.global_TSG[src, dst]['adjacency_matrix'] # A matrix in the paper
    #     spectrum_distribution, slots_per_path = self.get_spectrum_distribution_features(self.k_shortest_paths, self.current_service)

    #     # Combine into observation
    #     observation = {
    #         'source_one_hot': source_one_hot,
    #         'dest_one_hot': dest_one_hot,
    #         'slots_per_path': np.array(slots_per_path),
    #         'spectrum_features': np.array(spectrum_distribution),
    #         'feature_matrix': feature_matrix,
    #         'adjacency_matrix':adjacency_matrix
    #     }

    def observation(self):
        """
        Return flattened observation reshaped similar to original DeepRMSA.
        """
        src = self.current_service.source
        dst = self.current_service.destination

        # 1. Source-destination encoding [2, num_nodes]
        source_one_hot = np.zeros(self.topology.number_of_nodes())
        dest_one_hot = np.zeros(self.topology.number_of_nodes())
        source_one_hot[self.topology.nodes[src]['index']] = 1
        dest_one_hot[self.topology.nodes[dst]['index']] = 1
        
        # Stack source and dest to create source_destination_tau
        source_destination_tau = np.vstack([source_one_hot, dest_one_hot])  # Shape: [2, num_nodes]
        # Stack and reshape source-dest
        source_dest_tau = np.vstack([source_one_hot, dest_one_hot])
        source_dest_reshaped = source_dest_tau.reshape((1, np.prod(source_dest_tau.shape)))
        print(f"\nSource-dest shape: {source_dest_reshaped.shape}, size: {np.prod(source_dest_reshaped.shape)}")


        # 2. Get GCN inputs from global_TSG
        feature_matrix = self.global_TSG[(src, dst)]['feature_matrix']      # Shape: [num_nodes, k_paths]
        adjacency_matrix = self.global_TSG[(src, dst)]['adjacency_matrix']  # Shape: [num_nodes, num_nodes]
        feature_matrix_reshaped = feature_matrix.reshape((1, np.prod(feature_matrix.shape)))
        adjacency_matrix_reshaped = adjacency_matrix.reshape((1, np.prod(adjacency_matrix.shape)))
        print("Raw adj matrix Shape", adjacency_matrix.shape)
        print(f"Feature matrix shape: {feature_matrix_reshaped.shape}, size: {np.prod(feature_matrix_reshaped.shape)}")
        print(f"Adjacency matrix shape: {adjacency_matrix_reshaped.shape}, size: {np.prod(adjacency_matrix_reshaped.shape)}")

        # 3. Get spectrum distribution and slots
        spectrum_dict, slots_per_path = self.get_spectrum_distribution_features(
            self.k_shortest_paths, self.current_service)
        
        # Convert spectrum dictionary to array
        # Each path has 6 features per band
        spectrum_array = []
        for path_idx in range(len(slots_per_path)):
            path_features = []
            for band in range(self.num_bands):
                features = [
                    spectrum_dict[path_idx][band]['N_FS'],
                    spectrum_dict[path_idx][band]['N_FSB'],
                    spectrum_dict[path_idx][band]['N_FSB_prime'],
                    spectrum_dict[path_idx][band]['I_start'] if spectrum_dict[path_idx][band]['I_start'] != -1 else 0,
                    spectrum_dict[path_idx][band]['S_first'] if spectrum_dict[path_idx][band]['S_first'] != -1 else 0,
                    spectrum_dict[path_idx][band]['S_FSB']
                ]
                path_features.extend(features)
            spectrum_array.append(path_features)
        
        spectrum_array = np.array(spectrum_array)
        spectrum_reshaped = spectrum_array.reshape((1, np.prod(spectrum_array.shape)))
        slots_reshaped = np.array(slots_per_path).reshape((1, len(slots_per_path)))

        print(f"Spectrum array shape: {spectrum_reshaped.shape}, size: {np.prod(spectrum_reshaped.shape)}")
        print(f"Slots shape: {slots_reshaped.shape}, size: {np.prod(slots_reshaped.shape)}")
        
        # 4. Reshape components
        #source_dest_reshaped = source_destination_tau.reshape((1, np.prod(source_destination_tau.shape)))
        # feature_matrix_reshaped = feature_matrix.reshape((1, np.prod(feature_matrix.shape)))
        # adjacency_matrix_reshaped = adjacency_matrix.reshape((1, np.prod(adjacency_matrix.shape)))
        

        # 5. Concatenate all reshaped components
        observation = np.concatenate(
            (
                source_dest_reshaped,        # [1, 2*num_nodes]
                slots_reshaped,              # [1, k_paths]
                spectrum_reshaped,           # [1, k_paths*6*num_bands]
                feature_matrix_reshaped,     # [1, num_nodes*k_paths]
                adjacency_matrix_reshaped    # [1, num_nodes*num_nodes]
            ), 
            axis=1
        ).reshape(self.observation_space.shape)
        print(observation.shape)
        return observation



    def get_spectrum_distribution_features(self, k_shortest_paths, service):
        """
        Get spectrum availability distribution for all candidate paths using RMSA env methods.
        
        Args:
            rmsa_env: RMSA environment instance
            k_shortest_paths: Dictionary of k-shortest paths
            service_source: Source node
            service_destination: Destination node
        
        Returns:
            Dictionary of spectrum features for each path
        """
        src = service.source
        dst = service.destination
        paths = k_shortest_paths[(src, dst)]
        slots_per_path = []
        spectrum_features = {} 
        for path_idx, path in enumerate(paths):
            # Get available slots for this path
            num_slots = self.get_number_slots(path, self.num_bands, 0, self.modulations)
            slots_per_path.append(num_slots)

            spectrum_features[path_idx] = {}
            for band in range(self.num_bands):
                available_slots = self.get_available_slots(path, band)         
                num_slots = self.get_number_slots(path, self.num_bands, band, self.modulations)
                initial_indices, lengths = self.get_available_blocks(path_idx, self.num_bands, band, self.modulations)
                # Calculate spectrum features
                N_FS = np.sum(available_slots)  # total available FSs
                N_FSB = len(lengths)  # total number of blocks
                
                # Count blocks that satisfy bandwidth requirement
                N_FSB_prime = len([l for l in lengths if l >= num_slots])   
                # Get first fit block information that satisfies bandwidth requirement
                for i, (start, length) in enumerate(zip(initial_indices, lengths)):
                    if length >= num_slots:  # Check if block is large enough
                        I_start = start      # Starting index of first valid block
                        S_first = length     # Size of first valid block
                        break
                else:  # No blocks satisfy bandwidth requirement
                    I_start = -1  
                    S_first = -1        
                # Calculate average block size
                S_FSB = np.mean(lengths) if lengths else 0   
                spectrum_features[path_idx][band] = {
                    'N_FS': N_FS,
                    'N_FSB': N_FSB,
                    'N_FSB_prime': N_FSB_prime,
                    'I_start': I_start,
                    'S_first': S_first,
                    'S_FSB': S_FSB,
                }
        return spectrum_features, slots_per_path

    

    def reward(self, band, path_selected):
        return 1 if self.current_service.accepted else -1

    def reset(self, only_episode_counters=True):
        return super().reset(only_episode_counters=only_episode_counters)

    def _get_route_block_id(self, action: int) -> Tuple[int, int]:
        route = action // (self.j * self.num_bands)
        band  = action // (self.j * self.k_paths)
        block = action % self.j
        return route, band, block


def shortest_path_first_fit(env: DeepRMSAEnv) -> int:
    if not env.allow_rejection:
        return 0
    else:
        initial_indices, _ = env.get_available_blocks(0)
        if len(initial_indices) > 0:  # if there are available slots
            return 0
        else:
            return env.k_paths * env.j


def shortest_available_path_first_fit(env: DeepRMSAEnv) -> int:
    for idp, _ in enumerate(
        env.k_shortest_paths[
            env.current_service.source, env.current_service.destination
        ]
    ):
        initial_indices, _ = env.get_available_blocks(idp)
        if len(initial_indices) > 0:  # if there are available slots
            return idp * env.j  # this path uses the first one
    return env.k_paths * env.j
