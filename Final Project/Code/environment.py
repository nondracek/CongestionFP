# Environment For Project #

import numpy as np
from simulation import CongestionGame

class Environment:
    def __init__(self):
        self.edges = [
            (50, 48, 1),
            (50, 246, 1),
            (48, 68, 1),
            (48, 100, 1),
            (48, 230, 1),
            (48, 163, 1),
            (163, 230, 1),
            (163, 161, 1),
            (163, 162, 1),
            (162, 161, 1),
            (162, 170, 1),
            (162, 233, 1),
            (162, 229, 1),
            (229, 233, 1),
            (246, 68, 1),
            (68, 100, 1),
            (68, 186, 1),
            (68, 90, 1),
            (68, 249, 1),
            (68, 158,1),
            (100, 230, 1),
            (100, 161, 1),
            (100, 164, 1),
            (100, 186, 1),
            (230, 161, 1),
            (161, 170, 1),
            (161, 164, 1),
            (170, 233, 1),
            (170, 137, 1),
            (170, 107, 1),
            (170,164, 1),
            (233, 137, 1),
            (158, 246, 1),
            (158, 125, 1),
            (249, 125, 1),
            (249, 90, 1),
            (249, 234, 1),
            (90, 186, 1),
            (186, 164, 1),
            (186, 234, 1),
            (164, 234, 1),
            (164, 107, 1),
            (107, 137, 1),
            (107, 224, 1),
            (107, 79, 1),
            (107, 234, 1),
            (137, 224, 1),
            (125, 114, 1),
            (125, 211, 1),
            (125, 231, 1),
            (90, 113, 1),
            (234, 79, 1),
            (114, 113, 1),
            (114, 211, 1),
            (114, 144, 1),
            (114, 148, 1),
            (114, 79, 1),
            (113, 79, 1),
            (79, 148, 1),
            (79, 4, 1),
            (79, 224, 1),
            (224, 4, 1),
            (148, 4, 1),
            (148, 232, 1),
            (148, 144, 1),
            (148, 45, 1),
            (231, 211, 1),
            (231, 144, 1),
            (231, 45, 1),
            (231, 209, 1),
            (231, 261, 1),
            (231, 13, 1),
            (211, 144, 1),
            (144, 45, 1),
            (4, 232, 1),
            (45, 232, 1),
            (45, 209, 1),
            (13, 261, 1),
            (13, 12, 1),
            (12, 261, 1),
            (12, 88, 1),
            (261, 88, 1),
            (261, 87, 1),
            (261, 209, 1),
            (88, 87, 1),
            (87, 209, 1)
        ]
        self.weights = np.ones(len(self.edges))
        self.state_space_n = len(self.edges)
        self.action_space_n = 2*len(self.edges) + 1
        self.weight_max = 10
        self.weight_min = 1
        self.increment_val = 10
        self.traffic_sim = CongestionGame('nyc_trips.csv')

    def step(self, action):
        next_state = self.weights
        done = False

        if action == 0:
            done = True
        else:
            street_change = int(action/3)
            change = action % 2

            if change == 1:
                next_state[street_change] += self.increment_val
            elif change == 0 and (next_state[street_change] >= (self.weight_min + self.increment_val)):
                next_state[street_change] -= self.increment_val

        reward, congestion = self.get_reward(next_state)
        self.weights = next_state

        return next_state, reward, congestion, done

    def get_reward(self, state):
        new_edges = []
        for i in range(len(self.edges)):
            new_edges.append((self.edges[i][0], self.edges[i][1], self.weights[i]))

        self.edges = new_edges

        return self.traffic_sim.get_reward(new_edges)

    def reset(self):
        self.weights = np.ones(len(self.edges))
        return self.weights
