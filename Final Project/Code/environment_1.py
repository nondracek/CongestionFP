# Environment For Project #

import numpy as np
from collections import defaultdict
import collections
import multiprocessing as mp
from scipy import stats
import pandas as pd
import itertools
from simulation import CongestionGame
from simulation import Graph


def dijsktra(graph, initial, end):
# shortest paths is a dict of nodes
# whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            print(initial)
            print(end)
            return "Route Not Possible"
          # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
      # Reverse path
    path = path[::-1]
    return path

agent_csv = "nyc_trips.csv"
agents = pd.read_csv(agent_csv, index_col=0)
agents_dict = {(key1, key2): value for key1, key2, value in np.array(agents)}
agents_list = list(agents_dict.items())
agents_dict= list(map(list, agents_dict.items()))
agents = agents.reset_index()

def simulation(initial_edges):
    edge_dictionary = {}
    for edge in initial_edges:
        s, f, w = edge
        edge_dictionary[(s,f)] = w

    graph = Graph()
    for (s,f) , w in edge_dictionary.items():
        graph.add_edge(s,f,w)

    counter = 0
    optimals = []
    total_values = []
    changes = [1]
    iterations = []
    while changes[-1] != 0:
        counter += 1
        optimal = {}
        ordering = np.random.choice(len(agents_list),len(agents_list), replace=False)
        for a, step in enumerate(ordering):
            (po, do), _ = agents_dict[step]
            if len(optimals) > 0:
                for i in range(len(optimals[-1][step]) - 1):
                    graph.weights[(optimals[-1][step][i], optimals[-1][step][i+1])] -= agents_dict[step][1]
            optimal[step] = dijsktra(graph, po, do)
            for i in range(len(optimal[step]) - 1):
                graph.weights[(optimal[step][i], optimal[step][i+1])] += agents_dict[step][1]

        if len(optimals) < 2:
            optimals += [optimal]
        elif len(optimals) == 2:
            optimals = [optimals[1]]
            optimals += [optimal]
            a = list(map(lambda x: list(collections.OrderedDict(sorted(x.items())).values()), optimals))
            change = 0
            for j, route in enumerate(a[0]):
                if route != a[1][j]:
                    change +=1
            changes += [change]
        elif len(optimals) > 2:
            print("error")

        iterations += [optimal]
        total_values += [sum(list(graph.weights.values()))]
    return optimals[-1]


def get_equilibrium(initial_edges):
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(simulation, (initial_edges,)) for i in range((100 // mp.cpu_count())*mp.cpu_count())]
    results = [result.get() for result in results]
    pool.close()
    pool.join()
    equilibrium = results
    equilibrium = list(map(lambda x: list(collections.OrderedDict(sorted(x.items())).values()), equilibrium))
    new_equilibrium = []
    for sim in equilibrium:
        new_sim = []
        for path in sim:
            new_sim += [",".join([str(num) for num in path])]
        new_equilibrium += [new_sim]
    final_equilibirum = np.ndarray.flatten((stats.mode(np.transpose(np.array(new_equilibrium)), axis=1)[0]))
    return final_equilibirum

def get_reward_sim(initial_edges):
    eq = get_equilibrium(initial_edges)
    eq = [list(map(int,i.split(","))) for i in eq]
    graph = Graph()
    edge_dictionary = {}
    for edge in initial_edges:
        s, f, w = edge
        edge_dictionary[(s,f)] = w

    for (s,f) , w in edge_dictionary.items():
        graph.add_edge(s,f,w)

    initial_weights = graph.weights.copy()

    for i, route in enumerate(eq):
        for j in range(len(route) - 1):
            graph.weights[(route[j], route[j+1])] += agents_dict[i][1]

    cost_weights = {key: graph.weights[key] - initial_weights.get(key, 0) for key in graph.weights.keys()}

    reward = 0
    for i, route in enumerate(eq):
        for j in range(len(route) - 1):
            reward += (cost_weights[(route[j], route[j+1])] * agents_dict[i][1])

    

    return (reward * -1), eq



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
        self.increment_val = 10000
        self.increment_decay = .99

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
        self.increment_val = int(self.increment_val * self.increment_decay)

        return next_state, reward, congestion, done

    def get_reward(self, state):
        new_edges = []
        for i in range(len(self.edges)):
            new_edges.append((self.edges[i][0], self.edges[i][1], self.weights[i]))

        self.edges = new_edges

        return get_reward_sim(new_edges)

    def reset(self):
        self.weights = np.ones(len(self.edges))
        self.increment_val = 10000
        return self.weights
