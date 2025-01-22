import copy
import heapq
import random
from typing import List, Optional, Tuple

import gym
import networkx as nx
import numpy as np

from optical_rl_gym.utils import Service
from optical_rl_gym.utils import Service, Path, get_k_shortest_paths, get_path_weight




class OpticalNetworkEnv(gym.Env):
    def __init__(
        self,
        topology: nx.Graph = None,
        episode_length: int = 1000,
        load: float = 10.0,
        mean_service_holding_time: float = 10800.0,
        num_spectrum_resources: int = 80,
        allow_rejection: bool = False,
        node_request_probabilities: Optional[np.array] = None,
        seed: Optional[int] = None,
        channel_width: float = 12.5,
        k_paths=5
    ):
        assert topology is None or "ksp" in topology.graph
        assert topology is None or "k_paths" in topology.graph

        

        self._events: List[Tuple[float, Service]] = []
        self.current_time: float = 0
        self.episode_length: int = episode_length
        self.services_processed: int = 0
        self.services_accepted: int = 0
        self.episode_services_processed: int = 0
        self.episode_services_accepted: int = 0

        self.current_service: Service = None
        self._new_service: bool = False
        self.allow_rejection: bool = allow_rejection

        self.load: float = 0
        self.mean_service_holding_time: float = 0
        self.mean_service_inter_arrival_time: float = 0
        self.set_load(load=load, mean_service_holding_time=mean_service_holding_time)

        self.rand_seed: Optional[int] = None
        self.rng: random.Random = None
        self.seed(seed=seed)

        #self.topology: nx.Graph = copy.deepcopy(topology)
        self.topology: nx.DiGraph = copy.deepcopy(topology)
        #self.topology = self._convert_to_directed(topology)
        self.transformed_topo=nx.line_graph(self.topology)
        self.edge_to_index = {(u, v): self.topology[u][v]['index'] for u, v in self.topology.edges()}
        # Relabel nodes in transformed graph from edge tuples to indices
        self.index_mapping = {edge: self.edge_to_index[edge] for edge in self.transformed_topo.nodes()}
        self.transformed = nx.relabel_nodes(self.transformed_topo, self.index_mapping)
        self.topology_name: str = topology.graph["name"]
        self.k_paths: int = self.topology.graph["k_paths"]
        # just as a more convenient way to access it
        self.k_shortest_paths = self.topology.graph["ksp"]
        self.global_TSG = self.construct_tsg()

        assert (
            node_request_probabilities is None
            or len(node_request_probabilities) == self.topology.number_of_nodes()
        )
        self.num_spectrum_resources: int = num_spectrum_resources

        # channel width in GHz
        self.channel_width: float = channel_width
        self.topology.graph["num_spectrum_resources"] = num_spectrum_resources
        self.topology.graph["available_spectrum"] = np.full(
            (self.topology.number_of_edges()),
            fill_value=self.num_spectrum_resources,
            dtype=int,
        )
        if node_request_probabilities is not None:
            self.node_request_probabilities = node_request_probabilities
        else:
            self.node_request_probabilities = np.full(
                (self.topology.number_of_nodes()), fill_value=1.0 / self.topology.number_of_nodes())

        print(self.topology.edges)    

    def construct_tsg(self) -> dict:
        """
        Construct TSG from pre-transformed topology for all source-destination pairs.
        
        Args:
            topology: Original topology
            transformed_topo: Pre-transformed topology (line graph)
            k_shortest_paths: Dictionary containing k-shortest paths
            
        Returns:
            dict: Dictionary containing TSG, adjacency matrix and feature matrix for each s-d pair
        """
        global_TSG = {}
        N = len(self.transformed.nodes())  # Total number of nodes in transformed topology
        
        # Process each source-destination pair
        for (src, dst), paths in self.k_shortest_paths.items():
            #print(f"\nProcessing source {src} to destination {dst}")
            
            # Collect all link indices used in these paths
            link_indices = set()
            for path in paths:
                link_indices.update(path.link_idx)
            
            # Create subgraph with only the nodes (transformed from links) we need
            tsg = self.transformed.subgraph(link_indices).copy()
            
            # Add path membership information to nodes
            for node in tsg.nodes():
                path_indices = []
                for path_idx, path in enumerate(paths):
                    if node in path.link_idx:
                        path_indices.append(path_idx)
                tsg.nodes[node]['path_indices'] = path_indices
            
            # Create adjacency matrix
            adjacency_matrix = np.zeros((N, N))
            for path in paths:
                for i in range(len(path.link_idx) - 1):
                    current_link = path.link_idx[i]
                    next_link = path.link_idx[i + 1]
                    adjacency_matrix[current_link][next_link] = 1
            
            # Create feature matrix
            num_paths = len(paths)
            X = np.zeros((N, num_paths))
            for path_idx, path in enumerate(paths):
                for link_idx in path.link_idx:
                    X[link_idx][path_idx] = 1
            
            # Add graph level information
            tsg.graph['paths'] = paths
            tsg.graph['k_paths'] = len(paths)
            tsg.graph['source'] = src
            tsg.graph['destination'] = dst
            
            # Store in global_TSG dictionary
            global_TSG[(src, dst)] = {
                'tsg': tsg,
                'adjacency_matrix': adjacency_matrix,
                'feature_matrix': X,
                'num_paths': num_paths,
                'paths': paths
            }
            
            # Print summary for this s-d pair
            # print(f"Number of paths: {num_paths}")
            # print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
            # print(f"Feature matrix shape: {X.shape}")
            
        return global_TSG

    def set_load(self, load: float = None, mean_service_holding_time: float = None) -> None:
        """
        Sets the load to be used to generate requests.
        :param load: The load to be generated, in Erlangs
        :param mean_service_holding_time: The mean service holding time to be used to
        generate the requests
        :return: None
        """
        if load is not None:
            self.load = load
        if mean_service_holding_time is not None:
            self.mean_service_holding_time = (
                mean_service_holding_time  # current_service holding time in seconds
            )
        self.mean_service_inter_arrival_time = 1 / float(
            self.load / float(self.mean_service_holding_time)
        )

    def _plot_topology_graph(self, ax) -> None:
        pos = nx.get_node_attributes(self.topology, "pos")
        nx.draw_networkx_edges(self.topology, pos, ax=ax)
        nx.draw_networkx_nodes(
            self.topology,
            pos,
            nodelist=[
                x
                for x in self.topology.nodes()
                if x in [self.current_service.source, self.current_service.destination]
            ],
            label=[x for x in self.topology.nodes()],
            node_shape="s",
            node_color="white",
            edgecolors="black",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            self.topology,
            pos,
            nodelist=[
                x
                for x in self.topology.nodes()
                if x
                not in [self.current_service.source, self.current_service.destination]
            ],
            label=[x for x in self.topology.nodes()],
            node_shape="o",
            node_color="white",
            edgecolors="black",
            ax=ax,
        )
        nx.draw_networkx_labels(self.topology, pos)
        nx.draw_networkx_edge_labels(
            self.topology,
            pos,
            edge_labels={
                (i, j): "{}".format(
                    self.available_spectrum[self.topology[i][j]["index"]]
                )
                for i, j in self.topology.edges()
            },
        )
        # TODO: implement a trigger (a flag) that tells whether to plot the edge labels
        # set also an universal label dictionary inside the edge dictionary, e.g.,
        # (self.topology[a][b]['plot_label']

    def _add_release(self, service: Service) -> None:
        heapq.heappush(self._events, (service.arrival_time + service.holding_time, service))

    def _get_node_pair(self) -> Tuple[str, int, str, int]:
        src = self.rng.choices([x for x in self.topology.nodes()], weights=self.node_request_probabilities)[0]
        src_id = self.topology.graph["node_indices"].index(src)
        new_node_probabilities = np.copy(self.node_request_probabilities)
        new_node_probabilities[src_id] = 0.0
        new_node_probabilities = new_node_probabilities / np.sum(new_node_probabilities)
        dst = self.rng.choices([x for x in self.topology.nodes()], weights=new_node_probabilities)[0]
        dst_id = self.topology.graph["node_indices"].index(dst)
        return src, src_id, dst, dst_id

    def observation(self):
        return {"topology": self.topology, "service": self.current_service}

    def reward(self):
        return 1 if self.current_service.accepted else -1

    def reset(self) -> None:
        self._events = []
        self.current_time = 0
        self.services_processed = 0
        self.services_accepted = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0

        self.topology.graph["available_spectrum"] = np.full(
            self.topology.number_of_edges(),
            fill_value=self.num_spectrum_resources,
            dtype=int,
        )

        self.topology.graph["services"] = []
        self.topology.graph["running_services"] = []

        self.topology.graph["last_update"] = 0.0
        for lnk in self.topology.edges():
            self.topology[lnk[0]][lnk[1]]["utilization"] = 0.0
            self.topology[lnk[0]][lnk[1]]["last_update"] = 0.0
            self.topology[lnk[0]][lnk[1]]["services"] = []
            self.topology[lnk[0]][lnk[1]]["running_services"] = []

    def seed(self, seed=None):
        if seed is not None:
            self.rand_seed = seed
        else:
            self.rand_seed = 41
        self.rng = random.Random(self.rand_seed)
