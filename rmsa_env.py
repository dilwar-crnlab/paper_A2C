import copy
import functools
import heapq
import logging
import math
from collections import defaultdict
from typing import Optional, Sequence, Tuple

import gym
import networkx as nx
import numpy as np
import random
from optical_rl_gym.utils import Path, Service
from optical_rl_gym.osnr_calculator import *

from .optical_network_env import OpticalNetworkEnv


class RMSAEnv(OpticalNetworkEnv):

    metadata = {
        "metrics": [
            "service_blocking_rate",
            "episode_service_blocking_rate",
            "bit_rate_blocking_rate",
            "episode_bit_rate_blocking_rate",
        ]
    }

    def __init__(
        self,
        num_bands=None,
        topology: nx.Graph = None,
        episode_length: int = 1000,
        load: float = 10,
        mean_service_holding_time: float = 10800.0,
        #num_spectrum_resources: int = 100,
        #bit_rate_selection: str = "discrete",
        bit_rates: Sequence = [10, 40, 100],
        bit_rate_probabilities: Optional[np.array] = None,
        node_request_probabilities: Optional[np.array] = None,
        #bit_rate_lower_bound: float = 25.0,
        #bit_rate_higher_bound: float = 100.0,
        seed: Optional[int] = None,
        allow_rejection: bool = False,
        reset: bool = True,
        channel_width: float = 12.5,
        k_paths=5
    ):
        super().__init__(
            topology,
            episode_length=episode_length,
            load=load,
            mean_service_holding_time=mean_service_holding_time,
            #num_spectrum_resources=num_spectrum_resources,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            allow_rejection=allow_rejection,
            channel_width=channel_width,
            k_paths=k_paths
        )

        # make sure that modulations are set in the topology
        #assert "modulations" in self.topology.graph

    
        self.physical_params = PhysicalParameters() # for using PhysicalParameters data class
        # Initialize OSNR calculator
        self.osnr_calculator = OSNRCalculator()
        self.num_bands = num_bands
        # specific attributes for elastic optical networks
        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0

        


        multi_band_spectrum_resources = [100, 256] #332 -> 100, 916->256
        if self.num_bands == 1:
            self.num_spectrum_resources = multi_band_spectrum_resources[0]
        elif self.num_bands == 2:
            self.num_spectrum_resources = multi_band_spectrum_resources[1]

        #bit error rate (BER) of 10−3 are 9 dB, 12dB, 16 dB, and 18.6 dB,
        self.OSNR_th ={
            'BPSK': 9,
            'QPSK': 12,
            '8QAM': 16,
            '16QAM': 18.6
        }
        # Frequency ranges for C and L bands (in THz)
        self.band_frequencies = {
            0: {  # C-band
                'start': 191.3e12,  # Hz
                'end': 196.08e12,    # THz
            },
            1: {  # L-band
                'start': 184.4e12,  # THz
                'end': 191.3e12,    # THz
            }
        }



        self.spectrum_usage = np.zeros(
            (self.topology.number_of_edges() * self.num_bands, self.num_spectrum_resources), dtype=int
        )

        self.spectrum_slots_allocation = np.full(
            (self.topology.number_of_edges() * self.num_bands, self.num_spectrum_resources),
            fill_value=-1,
            dtype=int,
        )

        # do we allow proactive rejection or not?
        self.reject_action = 1 if allow_rejection else 0

        # defining the observation and action spaces
        self.actions_output = np.zeros((self.k_paths + 1, 
                                        self.num_bands + 1,
                                        self.num_spectrum_resources + 1), dtype=int
        )
        self.episode_actions_output = np.zeros(
            (self.k_paths + 1, self.num_bands + 1, self.num_spectrum_resources + 1), dtype=int
        )
        self.actions_taken = np.zeros(
            (self.k_paths + 1, self.num_bands + 1, self.num_spectrum_resources + 1), dtype=int
        )
        self.episode_actions_taken = np.zeros(
            (self.k_paths + 1, self.num_bands + 1, self.num_spectrum_resources + 1), dtype=int
        )
        self.action_space = gym.spaces.MultiDiscrete(
            (
                self.k_paths + self.reject_action,
                self.num_bands + self.reject_action,
                self.num_spectrum_resources + self.reject_action,
            )
        )
        self.observation_space = gym.spaces.Dict(
            {
                "topology": gym.spaces.Discrete(10),
                "current_service": gym.spaces.Discrete(10),
            }
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.logger = logging.getLogger("rmsaenv")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                "Logging is enabled for DEBUG which generates a large number of messages. "
                "Set it to INFO if DEBUG is not necessary."
            )
        self._new_service = False
        if reset:
            self.reset(only_episode_counters=False)

    def checkOSNR(self, service: Service, topology):
        service_OSNR = OSNRCalculator.calculate_osnr(service, topology)

    def _calculate_center_frequency(self, service: Service):
        """
        Calculate center frequency for an allocation
        Args:
            Service
        Returns:
            Center frequency in THz
        """
        # Get band frequency range
        service_end_idx= service.initial_slot + service.number_slots -1
        service_center = (service.initial_slot + service_end_idx)/2
        if service.band == 0:
            center_freq = self.band_frequencies[service.band]['start'] + (service_center * 12.5e9)
        elif service.band == 1:
            center_freq = self.band_frequencies[service.band]['start'] + (service_center - 100 ) * 12.5e9
        
        return center_freq

    def step(self, action: [int]):
        path, band, initial_slot = action[0], action[1], action[2]

        # registering overall statistics
        self.actions_output[path, band, initial_slot] += 1
        previous_network_compactness = (
            self._get_network_compactness()
        )  # used for compactness difference measure

        # starting the service as rejected
        self.current_service.accepted = False
        if (path < self.k_paths and band < self.num_bands and initial_slot < self.num_spectrum_resources):  # action is for assigning a path
            temp_path = self.k_shortest_paths[self.current_service.source, self.current_service.destination][path]
            if temp_path.length <= 4000:
                #print("Temp path len", temp_path.length )

                slots = self.get_number_slots(self.k_shortest_paths[self.current_service.source, self.current_service.destination][path], self.num_bands, band, self.modulations)
                self.logger.debug(
                    "{} processing action {} path {} and initial slot {} for {} slots".format(
                        self.current_service.service_id, action, path, initial_slot, slots))
                if self.is_path_free(self.k_shortest_paths[self.current_service.source, self.current_service.destination][path], initial_slot, slots, band ):
                        
                        #check for OSNR
                        temp_service = copy.deepcopy(self.current_service)
                        temp_service.bandwidth = slots * 12.5e9 # in GHz
                        temp_service.band = band
                        temp_service.initial_slot = initial_slot
                        temp_service.number_slots = slots
                        temp_service.path = self.k_shortest_paths[self.current_service.source, self.current_service.destination][path]

                        temp_service.center_frequency = self._calculate_center_frequency(temp_service)

                        modulation = self.get_modulation_format(temp_path, self.num_bands, band, self.modulations)['modulation']
                        temp_service.modulation_format = modulation
                        
                        #print("Temp serive:", temp_service)
                        osnr_db = self.osnr_calculator.calculate_osnr(temp_service, self.topology)
                        #print("OSNR", osnr_db)
                        if osnr_db >= self.OSNR_th[temp_service.modulation_format]:
                            #print("OSNR", osnr)
                            self.current_service.current_OSNR = osnr_db
                            self.current_service.OSNR_th = self.OSNR_th[temp_service.modulation_format]       
                            # if so, provision it (write zeros the position os the selected block in the available slots matrix
                            self._provision_path(self.k_shortest_paths[self.current_service.source, self.current_service.destination][path],
                                                initial_slot, slots, band, self.current_service.arrival_time)
                            self.current_service.accepted = True  # the request was accepted
                            self.current_service.modulation_format = modulation
                            self.actions_taken[path, band, initial_slot] += 1
                            self._add_release(self.current_service)
                else:
                    self.current_service.accepted = False  # the request was rejected (blocked), the path is not free
        else:
            self.current_service.accepted = False # the request was rejected (blocked), the path is not free
                

        if not self.current_service.accepted:
            self.actions_taken[self.k_paths, self.num_bands, self.num_spectrum_resources] += 1

        self.topology.graph["services"].append(self.current_service)

        # generating statistics for the episode info
        

        cur_network_compactness = (self._get_network_compactness())  # measuring compactness after the provisioning
        k_paths = self.k_shortest_paths[self.current_service.source, self.current_service.destination]
        path_selected = k_paths[path] if path < self.k_paths else None
        reward = self.reward(band, path_selected)
        info = {
            "band": band if self.services_accepted else -1,
            "service_blocking_rate": (self.services_processed - self.services_accepted)
            / self.services_processed,
            "episode_service_blocking_rate": (
                self.episode_services_processed - self.episode_services_accepted
            )
            / self.episode_services_processed,
            "bit_rate_blocking_rate": (
                self.bit_rate_requested - self.bit_rate_provisioned
            )
            / self.bit_rate_requested,
            "episode_bit_rate_blocking_rate": (
                self.episode_bit_rate_requested - self.episode_bit_rate_provisioned
            )
            / self.episode_bit_rate_requested,
            "network_compactness": cur_network_compactness,
            "network_compactness_difference": previous_network_compactness
            - cur_network_compactness,
            "avg_link_compactness": np.mean(
                [
                    self.topology[lnk[0]][lnk[1]]["compactness"]
                    for lnk in self.topology.edges()
                ]
            ),
            "avg_link_utilization": np.mean(
                [
                    self.topology[lnk[0]][lnk[1]]["utilization"]
                    for lnk in self.topology.edges()
                ]
            ),
        }

        # informing the blocking rate per bit rate
        # sorting by the bit rate to match the previous computation
       

        self._new_service = False
        self._next_service()
        return (
            self.observation(),
            reward,
            self.episode_services_processed == self.episode_length,
            info,
        )
    def reward(self, band, path_selected):
        return super().reward()
    
    def reset(self, only_episode_counters=True):
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.episode_actions_output = np.zeros(
            (
                self.k_paths + self.reject_action,
                self.num_bands + self.reject_action,
                self.num_spectrum_resources + self.reject_action,
            ),
            dtype=int,
        )
        self.episode_actions_taken = np.zeros(
            (
                self.k_paths + self.reject_action,
                self.num_bands + self.reject_action,
                self.num_spectrum_resources + self.reject_action,
            ),
            dtype=int,
        )

        

        if only_episode_counters:
            if self._new_service:
                # initializing episode counters
                # note that when the environment is reset, the current service remains the same and should be accounted for
                self.episode_services_processed += 1
                self.episode_bit_rate_requested += self.current_service.bit_rate
                
            return self.observation()

        super().reset()

        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0

        self.topology.graph["available_slots"] = np.ones(
            (self.topology.number_of_edges() * self.num_bands, self.num_spectrum_resources), dtype=int
        )

        self.spectrum_slots_allocation = np.full(
            (self.topology.number_of_edges() * self.num_bands, self.num_spectrum_resources),
            fill_value=-1,
            dtype=int,
        )

        

        self.topology.graph["compactness"] = 0.0
        self.topology.graph["throughput"] = 0.0
        for lnk in self.topology.edges():
            self.topology[lnk[0]][lnk[1]]["external_fragmentation"] = 0.0
            self.topology[lnk[0]][lnk[1]]["compactness"] = 0.0

        self._new_service = False
        self._next_service()
        return self.observation()

    def render(self, mode="human"):
        return

    # def _provision_path(self, path: Path, initial_slot, number_slots, band, at):
    #     # usage
    #     if not self.is_path_free(path, initial_slot, number_slots, band):
    #         raise ValueError(
    #             "Path {} has not enough capacity on slots {}-{}".format(
    #                 path.node_list, path, initial_slot, initial_slot + number_slots
    #             )
    #         )

    #     self.logger.debug(
    #         "{} assigning path {} on initial slot {} for {} slots".format(
    #             self.current_service.service_id,
    #             path.node_list,
    #             initial_slot,
    #             number_slots,
    #         )
    #     )
    #     # computing horizontal shift in the available slot matrix
    #     x = self.get_shift(band)[0]
    #     initial_slot_shift = initial_slot + x
    #     for i in range(len(path.node_list) - 1):
    #         self.topology.graph["available_slots"][
    #             ((self.topology[path.node_list[i]][path.node_list[i + 1]]["index"]) +
    #             (self.topology.number_of_edges()* band)),
    #             initial_slot_shift : initial_slot_shift + number_slots,
    #         ] = 0
    #         self.spectrum_slots_allocation[
    #             ((self.topology[path.node_list[i]][path.node_list[i + 1]]["index"]) +
    #             (self.topology.number_of_edges()* band)),
    #             initial_slot_shift : initial_slot_shift + number_slots,
    #         ] = self.current_service.service_id

    #         self.topology[path.node_list[i]][path.node_list[i + 1]]["services"].append(self.current_service)
    #         self.topology[path.node_list[i]][path.node_list[i + 1]]["running_services"].append(self.current_service)
    #         self._update_link_stats(path.node_list[i], path.node_list[i + 1])
    #     self.topology.graph["running_services"].append(self.current_service)
    #     self.current_service.path = path
    #     self.current_service.band = band
    #     self.current_service.initial_slot = initial_slot_shift
    #     self.current_service.number_slots = number_slots
    #     self.current_service.bandwidth = number_slots * 12.5e9
    #     #self.service.modulation_format = 

    #     self.current_service.termination_time = self.current_time + at
    #     self.current_service.center_frequency = self._calculate_center_frequency(self.current_service) #Error with the band
    #     #self.service.accepted = True  # the request was accepted
    #     self._update_network_stats()

    #     self.services_accepted += 1
    #     self.episode_services_accepted += 1
    #     self.bit_rate_provisioned += self.current_service.bit_rate
    #     self.episode_bit_rate_provisioned += self.current_service.bit_rate

    def _provision_path(self, path: Path, initial_slot, number_slots, band, at):
        """Modified to handle directed links"""
        if not self.is_path_free(path, initial_slot, number_slots, band):
            raise ValueError("Path doesn't have enough capacity")

        # Computing horizontal shift
        x = self.get_shift(band)[0]
        initial_slot_shift = initial_slot + x
        
        # For directed path, we only update the links in the direction of the path
        for i in range(len(path.node_list) - 1):
            node1, node2 = path.node_list[i], path.node_list[i + 1]
            # Get index of directed link
            link_idx = self.topology[node1][node2]["index"]
            
            # Update spectrum availability
            self.topology.graph["available_slots"][
                link_idx + (self.topology.number_of_edges() * band),
                initial_slot_shift : initial_slot_shift + number_slots,
            ] = 0
            
            # Update spectrum allocation
            self.spectrum_slots_allocation[
                link_idx + (self.topology.number_of_edges() * band),
                initial_slot_shift : initial_slot_shift + number_slots,
            ] = self.current_service.service_id
            
            # Update service information
            self.topology[node1][node2]["services"].append(self.current_service)
            self.topology[node1][node2]["running_services"].append(self.current_service)
            
            # Update link statistics
            self._update_link_stats(node1, node2)
        
        # Update service and network information
        self.topology.graph["running_services"].append(self.current_service)
        self.current_service.path = path
        self.current_service.band = band
        self.current_service.initial_slot = initial_slot_shift
        self.current_service.number_slots = number_slots
        self.current_service.bandwidth = number_slots * 12.5e9
        self._update_network_stats()
        self.current_service.center_frequency = self._calculate_center_frequency(self.current_service)
        
        # Update counters
        self.services_accepted += 1
        self.episode_services_accepted += 1
        self.bit_rate_provisioned += self.current_service.bit_rate
        self.episode_bit_rate_provisioned += self.current_service.bit_rate

        



    def _release_path(self, service: Service):
        """Modified to handle directed links"""
        for i in range(len(service.path.node_list) - 1):
            node1, node2 = service.path.node_list[i], service.path.node_list[i + 1]
            # Get index of directed link
            link_idx = self.topology[node1][node2]["index"]
            
            # Release spectrum
            self.topology.graph["available_slots"][
                link_idx + (self.topology.number_of_edges() * service.band),
                service.initial_slot : service.initial_slot + service.number_slots
            ] = 1
            
            # Clear allocation
            self.spectrum_slots_allocation[
                link_idx + (self.topology.number_of_edges() * service.band),
                service.initial_slot : service.initial_slot + service.number_slots,
            ] = -1
            
            # Update service lists
            self.topology[node1][node2]["running_services"].remove(service)
            self._update_link_stats(node1, node2)
            
        self.topology.graph["running_services"].remove(service)

    def _update_network_stats(self):
        last_update = self.topology.graph["last_update"]
        time_diff = self.current_time - last_update
        if self.current_time > 0:
            last_throughput = self.topology.graph["throughput"]
            last_compactness = self.topology.graph["compactness"]

            cur_throughput = 0.0

            for service in self.topology.graph["running_services"]:
                cur_throughput += service.bit_rate

            throughput = (
                (last_throughput * last_update) + (cur_throughput * time_diff)
            ) / self.current_time
            self.topology.graph["throughput"] = throughput

            compactness = (
                (last_compactness * last_update)
                + (self._get_network_compactness() * time_diff)
            ) / self.current_time
            self.topology.graph["compactness"] = compactness

        self.topology.graph["last_update"] = self.current_time

    def _update_link_stats(self, node1: str, node2: str):
        last_update = self.topology[node1][node2]["last_update"]
        time_diff = self.current_time - self.topology[node1][node2]["last_update"]
        if self.current_time > 0:
            last_util = self.topology[node1][node2]["utilization"]
            cur_util = (
                self.num_spectrum_resources
                - np.sum(
                    self.topology.graph["available_slots"][
                        self.topology[node1][node2]["index"], :
                    ]
                )
            ) / self.num_spectrum_resources
            utilization = (
                (last_util * last_update) + (cur_util * time_diff)
            ) / self.current_time
            self.topology[node1][node2]["utilization"] = utilization

            slot_allocation = self.topology.graph["available_slots"][
                self.topology[node1][node2]["index"], :
            ]

            # implementing fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
            last_external_fragmentation = self.topology[node1][node2][
                "external_fragmentation"
            ]
            last_compactness = self.topology[node1][node2]["compactness"]

            cur_external_fragmentation = 0.0
            cur_link_compactness = 0.0
            if np.sum(slot_allocation) > 0:
                initial_indices, values, lengths = RMSAEnv.rle(slot_allocation)

                # computing external fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
                unused_blocks = [i for i, x in enumerate(values) if x == 1]
                max_empty = 0
                if len(unused_blocks) > 1 and unused_blocks != [0, len(values) - 1]:
                    max_empty = max(lengths[unused_blocks])
                cur_external_fragmentation = 1.0 - (
                    float(max_empty) / float(np.sum(slot_allocation))
                )

                # computing link spectrum compactness from https://ieeexplore.ieee.org/abstract/document/6421472
                used_blocks = [i for i, x in enumerate(values) if x == 0]

                if len(used_blocks) > 1:
                    lambda_min = initial_indices[used_blocks[0]]
                    lambda_max = (
                        initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]
                    )

                    # evaluate again only the "used part" of the spectrum
                    internal_idx, internal_values, internal_lengths = RMSAEnv.rle(
                        slot_allocation[lambda_min:lambda_max]
                    )
                    unused_spectrum_slots = np.sum(1 - internal_values)

                    if unused_spectrum_slots > 0:
                        cur_link_compactness = (
                            (lambda_max - lambda_min) / np.sum(1 - slot_allocation)
                        ) * (1 / unused_spectrum_slots)
                    else:
                        cur_link_compactness = 1.0
                else:
                    cur_link_compactness = 1.0

            external_fragmentation = (
                (last_external_fragmentation * last_update)
                + (cur_external_fragmentation * time_diff)
            ) / self.current_time
            self.topology[node1][node2][
                "external_fragmentation"
            ] = external_fragmentation

            link_compactness = (
                (last_compactness * last_update) + (cur_link_compactness * time_diff)
            ) / self.current_time
            self.topology[node1][node2]["compactness"] = link_compactness

        self.topology[node1][node2]["last_update"] = self.current_time

    def _next_service(self):
        if self._new_service:
            return
        at = self.current_time + self.rng.expovariate(
            1 / self.mean_service_inter_arrival_time
        )
        self.current_time = at

        ht = self.rng.expovariate(1 / self.mean_service_holding_time)
        src, src_id, dst, dst_id = self._get_node_pair()

        # generate the bit rate according to the selection adopted
        BitRate = [50, 100, 200]
        bit_rate = random.choice(BitRate)


        self.current_service = Service(
            self.episode_services_processed,
            src,
            src_id,
            destination=dst,
            destination_id=dst_id,
            arrival_time=at,
            holding_time=ht,
            bit_rate=bit_rate,
        )
        self._new_service = True

        self.services_processed += 1
        self.episode_services_processed += 1

        # registering statistics about the bit rate requested
        self.bit_rate_requested += self.current_service.bit_rate
        self.episode_bit_rate_requested += self.current_service.bit_rate
        #if self.bit_rate_selection == "discrete":
            #self.bit_rate_requested_histogram[bit_rate] += 1
            #self.episode_bit_rate_requested_histogram[bit_rate] += 1

            # we build the histogram of slots requested assuming the shortest path
            #slots = self.get_number_slots(self.k_shortest_paths[src, dst][0])
            #self.slots_requested_histogram[slots] += 1
            #self.episode_slots_requested_histogram[slots] += 1

        # release connections up to this point
        while len(self._events) > 0:
            (time, service_to_release) = heapq.heappop(self._events)
            if time <= self.current_time:
                self._release_path(service_to_release)
            else:  # release is not to be processed yet
                self._add_release(service_to_release)  # puts service back in the queue
                break  # breaks the loop

    def _get_path_slot_id(self, action: int) -> Tuple[int, int]:
        """
        Decodes the single action index into the path index and the slot index to be used.

        :param action: the single action index
        :return: path index and initial slot index encoded in the action
        """
        path = int(action / self.num_spectrum_resources)
        initial_slot = action % self.num_spectrum_resources
        return path, initial_slot

    def get_number_slots(self, path: Path, num_bands, band, modulations) -> int:
        """
        Method that computes the number of spectrum slots necessary to accommodate the service request into the path.
        The method already adds the guardband.
        """
        modulation = self.get_modulation_format(path, num_bands, band, modulations)
        service_bit_rate = self.current_service.bit_rate
        number_of_slots = math.ceil(service_bit_rate / modulation['capacity']) + 1
        return number_of_slots

    def get_shift(slef, band):
        x=0
        y=0
        if band==0:
            x=0
            y=100
        elif band==1:
            x=100
            y=256
        return x , y
    
    def is_path_free(self, path: Path, initial_slot: int, number_slots: int, band) -> bool:
        x = self.get_shift(band)[0]
        initial_slot_shift = initial_slot + x
        if initial_slot_shift + number_slots > self.num_spectrum_resources:
            # logging.debug('error index' + env.parameters.rsa_algorithm)
            return False
        for i in range(len(path.node_list) - 1):
            if np.any(self.topology.graph["available_slots"][
                    ((self.topology[path.node_list[i]][path.node_list[i + 1]]["index"]) +
                    (self.topology.number_of_edges() * band)),
                    initial_slot_shift : initial_slot_shift + number_slots] == 0):
                
                
            
                return False
        return True


    
    def get_available_slots(self, path: Path, band):
        """Modified to handle directed links"""
        x = self.get_shift(band)[0]
        y = self.get_shift(band)[1]
        
        # For directed path, only consider links in the path direction
        available_slots = functools.reduce(
            np.multiply,
            self.topology.graph["available_slots"][[((self.topology[path.node_list[i]][path.node_list[i + 1]]['index']) +
                (self.topology.number_of_edges() * band))
                for i in range(len(path.node_list) - 1)], x:y])
            
        
        return available_slots

    def rle(inarray):
        """run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values)"""
        # from: https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
        ia = np.asarray(inarray)  # force numpy
        n = len(ia)
        if n == 0:
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)  # must include last element posi
            z = np.diff(np.append(-1, i))  # run lengths
            p = np.cumsum(np.append(0, z))[:-1]  # positions
            return p, ia[i], z

    def get_available_blocks(self, path, num_bands, band, modulations):
        # get available slots across the whole path
        # 1 if slot is available across all the links
        # zero if not
        available_slots = self.get_available_slots(
            self.k_shortest_paths[
                self.current_service.source, self.current_service.destination
            ][path], band
        )

        # getting the number of slots necessary for this service across this path
        slots = self.get_number_slots(
            self.k_shortest_paths[
                self.current_service.source, self.current_service.destination
            ][path], num_bands, band, modulations
        )

        # getting the blocks
        initial_indices, values, lengths = RMSAEnv.rle(available_slots)

        # selecting the indices where the block is available, i.e., equals to one
        available_indices = np.where(values == 1)

        # selecting the indices where the block has sufficient slots
        sufficient_indices = np.where(lengths >= slots)

        # getting the intersection, i.e., indices where the slots are available in sufficient quantity
        # and using only the J first indices
        final_indices = np.intersect1d(available_indices, sufficient_indices)[: self.j]

        return initial_indices[final_indices], lengths[final_indices]

    def _get_network_compactness(self):
        # implementing network spectrum compactness from https://ieeexplore.ieee.org/abstract/document/6476152

        sum_slots_paths = 0  # this accounts for the sum of all Bi * Hi

        for service in self.topology.graph["running_services"]:
            sum_slots_paths += service.number_slots * service.path.hops

        # this accounts for the sum of used blocks, i.e.,
        # \sum_{j=1}^{M} (\lambda_{max}^j - \lambda_{min}^j)
        sum_occupied = 0

        # this accounts for the number of unused blocks \sum_{j=1}^{M} K_j
        sum_unused_spectrum_blocks = 0

        for n1, n2 in self.topology.edges():
            # getting the blocks
            initial_indices, values, lengths = RMSAEnv.rle(
                self.topology.graph["available_slots"][
                    self.topology[n1][n2]["index"], :
                ]
            )
            used_blocks = [i for i, x in enumerate(values) if x == 0]
            if len(used_blocks) > 1:
                lambda_min = initial_indices[used_blocks[0]]
                lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]
                sum_occupied += (
                    lambda_max - lambda_min
                )  # we do not put the "+1" because we use zero-indexed arrays

                # evaluate again only the "used part" of the spectrum
                internal_idx, internal_values, internal_lengths = RMSAEnv.rle(
                    self.topology.graph["available_slots"][
                        self.topology[n1][n2]["index"], lambda_min:lambda_max
                    ]
                )
                sum_unused_spectrum_blocks += np.sum(internal_values)

        if sum_unused_spectrum_blocks > 0:
            cur_spectrum_compactness = (sum_occupied / sum_slots_paths) * (
                self.topology.number_of_edges() / sum_unused_spectrum_blocks
            )
        else:
            cur_spectrum_compactness = 1.0

        return cur_spectrum_compactness

    def calculate_MF(self, modulations, length):
        for i in range(len(modulations) - 1):
            if length > modulations[i + 1]['max_reach']:
                if length <= modulations[i]['max_reach']:
                    return modulations[i]
        return modulations[len(modulations) - 1]
    
    def get_modulation_format(self, path: Path, num_bands, band, modulations):
        length= path.length
        if num_bands == 1: # C band
            modulation_format = self.calculate_MF(modulations, length)
        elif num_bands == 2: # C + L band
            if band == 0: # C band
                modulation_format = self.calculate_MF(modulations, length)
            elif band == 1: # L band
                modulation_format = self.calculate_MF(modulations, length)

        return modulation_format 

    '''
        Modluation format
    '''
    #[BPSK, QPSK, 8QAM, 16QAM]
    capacity = [12.5, 25, 37.5, 50]
    modulations = list()
    modulations.append({'modulation': 'BPSK', 'capacity': capacity[0], 'max_reach': 4000})
    modulations.append({'modulation': 'QPSK', 'capacity': capacity[1], 'max_reach': 2000})
    modulations.append({'modulation': '8QAM', 'capacity': capacity[2], 'max_reach': 1000})
    modulations.append({'modulation': '16QAM', 'capacity': capacity[3], 'max_reach': 500})




def shortest_path_first_fit(env: RMSAEnv) -> Tuple[int, int]:
    num_slots = env.get_number_slots(
        env.k_shortest_paths[
            env.current_service.source, env.current_service.destination
        ][0]
    )
    for initial_slot in range(
        0, env.topology.graph["num_spectrum_resources"] - num_slots
    ):
        if env.is_path_free(
            env.k_shortest_paths[
                env.current_service.source, env.current_service.destination
            ][0],
            initial_slot,
            num_slots,
        ):
            return (0, initial_slot)
    return (env.topology.graph["k_paths"], env.topology.graph["num_spectrum_resources"])


def shortest_available_path_first_fit(env: RMSAEnv) -> Tuple[int, int]:
    for idp, path in enumerate(
        env.k_shortest_paths[
            env.current_service.source, env.current_service.destination
        ]
    ):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(
            0, env.topology.graph["num_spectrum_resources"] - num_slots
        ):
            if env.is_path_free(path, initial_slot, num_slots):
                return (idp, initial_slot)
    return (env.topology.graph["k_paths"], env.topology.graph["num_spectrum_resources"])


def least_loaded_path_first_fit(env: RMSAEnv) -> Tuple[int, int]:
    max_free_slots = 0
    action = (
        env.topology.graph["k_paths"],
        env.topology.graph["num_spectrum_resources"],
    )
    for idp, path in enumerate(
        env.k_shortest_paths[
            env.current_service.source, env.current_service.destination
        ]
    ):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(
            0, env.topology.graph["num_spectrum_resources"] - num_slots
        ):
            if env.is_path_free(path, initial_slot, num_slots):
                free_slots = np.sum(env.get_available_slots(path))
                if free_slots > max_free_slots:
                    action = (idp, initial_slot)
                    max_free_slots = free_slots
                break  # breaks the loop for the initial slot
    return action


class SimpleMatrixObservation(gym.ObservationWrapper):
    def __init__(self, env: RMSAEnv):
        super().__init__(env)
        shape = (
            self.env.topology.number_of_nodes() * 2
            + self.env.topology.number_of_edges() * self.env.num_spectrum_resources
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=1, dtype=np.uint8, shape=(shape,)
        )
        self.action_space = env.action_space

    def observation(self, observation):
        source_destination_tau = np.zeros((2, self.env.topology.number_of_nodes()))
        min_node = min(
            self.env.current_service.source_id, self.env.current_service.destination_id
        )
        max_node = max(
            self.env.current_service.source_id, self.env.current_service.destination_id
        )
        source_destination_tau[0, min_node] = 1
        source_destination_tau[1, max_node] = 1
        spectrum_obs = copy.deepcopy(self.topology.graph["available_slots"])
        return np.concatenate(
            (
                source_destination_tau.reshape(
                    (1, np.prod(source_destination_tau.shape))
                ),
                spectrum_obs.reshape((1, np.prod(spectrum_obs.shape))),
            ),
            axis=1,
        ).reshape(self.observation_space.shape)


class PathOnlyFirstFitAction(gym.ActionWrapper):
    def __init__(self, env: RMSAEnv):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(
            self.env.k_paths + self.env.reject_action
        )
        self.observation_space = env.observation_space

    def action(self, action) -> Tuple[int, int]:
        if action < self.env.k_paths:
            num_slots = self.env.get_number_slots(
                self.env.k_shortest_paths[
                    self.env.current_service.source,
                    self.env.current_service.destination,
                ][action]
            )
            for initial_slot in range(
                0, self.env.topology.graph["num_spectrum_resources"] - num_slots
            ):
                if self.env.is_path_free(
                    self.env.k_shortest_paths[
                        self.env.current_service.source,
                        self.env.current_service.destination,
                    ][action],
                    initial_slot,
                    num_slots,
                ):
                    return (action, initial_slot)
        return (
            self.env.topology.graph["k_paths"],
            self.env.topology.graph["num_spectrum_resources"],
        )

    def step(self, action):
        return self.env.step(self.action(action))
