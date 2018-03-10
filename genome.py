import numpy as np
import networkx as nx
import random
from enum import Enum


#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# If it is not possible or desirable to put the notice in a particular
# file, then You may include the notice in a location (such as a LICENSE
# file in a relevant directory) where a recipient would be likely to look
# for such a notice.

# You may add additional accurate notices of copyright ownership.

# Written by Stefano Palmieri in December 2016


# This class stores the genome as a weighted Multi-Directed Acyclical
# Graph called an Compositional Adjacency Matrix Producing Network. The CAMPN
# graph is produced from Nodes and Links. Nodes can output a constant
# matrix (tensor) or perform a matrix operation such as Hadamard product.


# Enumerate node types
# If you want to add more node types, add after Hadamard
class NodeType(Enum):
    output = 0
    input = 1
    # actual matrix operations get placed under this line
    #############################################
    kronecker = 2
    hadamard = 3


class Genome(object):
    # Probability of mutating a new node
    P_NEW_NODE = 0.01

    # Probability of mutating a new input node
    P_NEW_INPUT_NODE = 0.01

    # Probability of mutating a new link
    P_NEW_LINK = 0.01

    # Probability of mutating an existing input node
    P_INPUT_NODE = 0.1

    # Probability of mutating existing link
    P_LINK = 0.3

    # Sizes for matrices in the input node type
    CONSTANT_ROW_SIZE = 4
    CONSTANT_COL_SIZE = 4

    # Mean and standard deviation for weight mutations
    MUTATE_MEAN = 0.1
    MUTATE_STD = 1.0

    # Constants and Coefficients for computing compatibility
    EXCESS_COEFFICIENT = 1.0
    DISJOINT_COEFFICIENT = 1.0
    WEIGHT_COEFFICIENT = 0.4

    def __init__(self, innovation_db=None):
        self.innovation_db = innovation_db
        self.network = nx.MultiDiGraph()

        # node with id 1 is the output
        self.network.add_node(1, type=NodeType.output)
        self.network.add_node(2, type=NodeType.input,
                              constant=np.matrix([[0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0]],
                                                 dtype=np.float16))
        self.network.add_edge(2, 1, key=True, weight=1.0, innovation=3)

        self.innovation_db.retrieve_innovation_num('output', 0, 0)
        self.innovation_db.retrieve_innovation_num('input', 0, 0)
        self.innovation_db.retrieve_innovation_num('left', 2, 1)

        self.fitness = None
        self.adjusted_fitness = None

    # Mutations can add a Node, add a Link, change the weight of a Link,
    # change the function of a Node.
    def mutate(self):

        # If mutation occurs, mutate a new node
        if random.random() <= self.P_NEW_NODE:
            self.mutate_new_node()

        # If mutation occurs, mutate a new input node
        if random.random() <= self.P_NEW_INPUT_NODE:
            self.mutate_new_input_node()

        # If mutation occurs, mutate an existing input node
        if random.random() <= self.P_INPUT_NODE:
            self.mutate_node()

        # If mutation occurs, mutate a new link
        if random.random() <= self.P_NEW_LINK:
            self.mutate_new_link()

        # If mutation occurs, mutate an existing link
        if random.random() <= self.P_LINK:
            self.mutate_link()

    # Create a new random Node by selecting a link and splitting it
    def mutate_new_node(self):

        # Randomly select a link from the existing network
        start, end, key = random.choice(self.network.edges(keys=True))

        # Randomly select the function for this node but don't include Input
        # or Output types as functions
        function = NodeType(random.randrange(NodeType.input.value + 1,
                                             len(NodeType)))

        # Create a new node with the function type
        node_num = self.innovation_db.retrieve_innovation_num(function.name,
                                                              start, end)
        self.network.add_node(node_num, type=function)

        # Add links for the new node
        in_key = random.random() < 0.5
        in_type = 'left' if in_key else 'right'
        in_innov_num = self.innovation_db.retrieve_innovation_num(in_type,
                                                                  start,
                                                                  node_num)
        self.network.add_edge(start, node_num,
                              key=in_key, weight=self.random_weight(),
                              innovation=in_innov_num)

        out_key = random.random() < 0.5
        out_type = 'left' if out_key else 'right'
        out_innov_num = self.innovation_db.retrieve_innovation_num(out_type,
                                                                   node_num,
                                                                   end)
        self.network.add_edge(node_num, end,
                              key=key, weight=self.random_weight(),
                              innovation=out_innov_num)

        # Remove old edge (in future versions this edge should
        # be disabled rather than removed)
        self.network.remove_edge(start, end, key=key)

    # Create a new random input Node by selecting a non-input node
    # and creating a child
    def mutate_new_input_node(self):

        # Get list of non-input nodes
        non_input_nodes = [node for node, data
                           in self.network.nodes(data=True)
                           if data['type'] != NodeType.input]

        # randomly pick a node from the list
        node = random.choice(non_input_nodes)

        # Create a new input node
        # node_num = self.network.number_of_nodes() + 1
        node_num = self.innovation_db.retrieve_innovation_num('input',
                                                              node, 0)
        self.network.add_node(node_num, type=NodeType.input,
                              constant=np.zeros((self.CONSTANT_ROW_SIZE,
                                                 self.CONSTANT_COL_SIZE),
                                                dtype=np.float16))

        # Create a link from the new input node to existing non-input node
        key = random.random() < 0.5
        link_type = 'left' if key else 'right'
        innov_num = self.innovation_db.retrieve_innovation_num(link_type,
                                                               node_num,
                                                               node)
        self.network.add_edge(node_num, node,
                              key=key,
                              weight=self.random_weight(),
                              innovation=innov_num)

    # Randomly mutates one of the existing input nodes in the CAMPN.
    def mutate_node(self):

        # get list of input nodes
        input_nodes = [node for node, data
                       in self.network.nodes(data=True)
                       if data['type'] == NodeType.input]

        # randomly pick a node from the list
        node = random.choice(input_nodes)

        # mutate one of it's constant's elements
        rand_row = random.randrange(0, self.CONSTANT_ROW_SIZE)
        rand_col = random.randrange(0, self.CONSTANT_COL_SIZE)

        # Mutate by Gaussian random variable
        change = np.random.normal(self.MUTATE_MEAN, self.MUTATE_STD)

        # apply the change
        self.network.node[node]['constant'][rand_row, rand_col] += change

    # Create a new Link and add it to the Genome.
    def mutate_new_link(self):
        # select a random node to start a link from and
        # don't include output node
        choices = [node for node, data
                   in self.network.nodes(data=True)
                   if data['type'] != NodeType.output]
        start = random.choice(choices)

        # select a random node to end link at, don't include input nodes
        choices = [node for node, data
                   in self.network.nodes(data=True)
                   if data['type'] != NodeType.input and node != start]
        end = random.choice(choices)

        # Create the link
        key = random.random() < 0.5
        link_type = 'left' if key else 'right'
        innov_num = self.innovation_db.retrieve_innovation_num(link_type,
                                                               start,
                                                               end)
        self.network.add_edge(start, end, key=key,
                              weight=self.random_weight(),
                              innovation=innov_num)

        # Check for no cycles. If there are, remove this link
        cycles = len(list(nx.simple_cycles(self.network)))

        if cycles:
            self.network.remove_edge(start, end, key=key)

    # Randomly mutate an existing link by modifying the weight
    def mutate_link(self):

        # Select a random link
        start, end, key = random.choice(self.network.edges(keys=True))

        # Add a Gaussian random variable to existing weight
        weight_change = np.random.normal(self.MUTATE_MEAN, self.MUTATE_STD)

        # Modify the weight on the link
        self.network[start][end][key]['weight'] += weight_change

    def random_weight(self):
        return np.random.normal(0.0, self.MUTATE_STD)

    # recursively compute output for a node
    def get_phenotype(self, node=None):

        # print("get phenotype: ", node)
        if node is None:
            return self.get_phenotype(1)

        if self.network.node[node]['type'] == NodeType.input:
            return self.network.node[node]['constant']

        left_logit = self.calculate_logit(node, True)
        right_logit = self.calculate_logit(node, False)

        return self.node_output(node, left_logit, right_logit)

    def calculate_logit(self, node, key):
        predecessors = list(self.network.predecessors(node))
        nodes = [predecessor for predecessor
                 in predecessors if key in self.network[predecessor][node]]

        if nodes:
            logit = np.zeros((self.CONSTANT_ROW_SIZE, self.CONSTANT_COL_SIZE))
            for x in nodes:
                weight = self.network[x][node][key]['weight']
                term = self.get_phenotype(x)
                # if it's a dangling (dead end node) just ignore it.
                if term is None:
                    continue
                logit, term = same_size(logit, term)
                logit += weight * term
            return logit
        # if there are no links for this logit, return false
        else:
            return None

    # Returns a recombinated genome taken between this organism and another
    def recombinate(self, other):
        offspring = Genome(self.innovation_db)

        for s, e, key, data in self.network.edges_iter(data=True, keys=True):
            innovation = data['innovation']
            added = False
            for d, r, key2, data2 in other.network.edges_iter(data=True, keys=True):
                innovation2 = data2['innovation']
                # randomly pick one of the genomes to inherit from
                if innovation == innovation2:
                    added = True
                    if random.random() < 0.5:
                        offspring.network.add_edge(s, e,
                                                   key=key,
                                                   weight=data['weight'],
                                                   innovation=data['innovation'])

                    else:
                        offspring.network.add_edge(d, r,
                                                   key=key2,
                                                   weight=data2['weight'],
                                                   innovation=data2['innovation'])

            if not added:
                offspring.network.add_edge(s, e,
                                           key=key,
                                           weight=data['weight'],
                                           innovation=data['innovation'])

        for s, e, key, data in other.network.edges_iter(data=True, keys=True):
            innovation = data['innovation']
            matched = False
            for d, r, key2, data2 in offspring.network.edges_iter(data=True, keys=True):
                innovation2 = data2['innovation']
                if innovation == innovation2:
                    matched = True
                    break

            if not matched:
                offspring.network.add_edge(s, e, key=key,
                                           weight=data['weight'],
                                           innovation=data['innovation'])

        for node, data in self.network.nodes_iter(data=True):
            offspring.network.add_node(node, data)

        for node, data in other.network.nodes_iter(data=True):
            offspring.network.add_node(node, data)

        # need to remove any cycles that created

        offspring.remove_cycles()
        '''
        try:
            offspring.get_phenotype()
        except IndexError:
            print("father is")
            for node in self.network.nodes(data=True):
                print(node)
            for edge in self.network.edges(data=True, keys=True):
                print(edge)
            print("mother is")
            for node in other.network.nodes(data=True):
                print(node)
            for edge in other.network.edges(data=True, keys=True):
                print(edge)
            exit()'''

        return offspring

    # Remove any cycles by randomly selecting links
    # within the cycle and removing them
    def remove_cycles(self):

        cycles = list(nx.simple_cycles(self.network))
        cycles_num = len(cycles)

        while cycles_num > 0:
            group = cycles[0]
            choice = random.randint(0, len(group)-1)
            end = group[choice]
            group.remove(end)
            start = group[choice-1]
            self.network.remove_edge(start, end)
            cycles = list(nx.simple_cycles(self.network))
            cycles_num = len(cycles)

    # Compute the distance using innovation numbers on the link genes
    def distance(self, other):
        innovations = self.get_link_innovations()
        other_innovations = other.get_link_innovations()

        innovations_set = set(innovations)
        other_innovations_set = set(other_innovations)

        last_disjoint = min(innovations[-1], other_innovations[-1])

        sym_dif = innovations_set ^ other_innovations_set
        disjoint = [elem for elem in sym_dif if elem <= last_disjoint]
        excess = [elem for elem in sym_dif if elem > last_disjoint]
        normalizer = max(len(innovations), len(other_innovations))

        disjoint_term = self.DISJOINT_COEFFICIENT * len(disjoint) / normalizer
        excess_term = self.EXCESS_COEFFICIENT * len(excess) / normalizer
        weight_term = self.WEIGHT_COEFFICIENT \
            * self.average_weight_difference(other)

        return disjoint_term + excess_term + weight_term

    def average_weight_difference(self, other):

        total_weight_difference = 0
        denominator = 0

        for s, e, data in self.network.edges_iter(data=True):
            weight = data['weight']
            innovation = data['innovation']
            for d, r, data2 in other.network.edges_iter(data=True):
                weight2 = data2['weight']
                innovation2 = data2['innovation']
                if innovation == innovation2:
                    total_weight_difference += abs(weight2 - weight)
                    denominator += 1
                    break
        if denominator == 0:
            return 0
        return total_weight_difference / denominator

    def get_link_innovations(self):
        innovations = [innovation for s, e, innovation
                       in self.network.edges_iter(data='innovation')]
        return innovations

    # compute output for the node
    def node_output(self, node, left_logit, right_logit):

        if left_logit is None and right_logit is None:
            return None

        try:
            # output node acts just like a sum
            if self.network.node[node]['type'] == NodeType.output:
                if left_logit is None:
                    return right_logit
                elif right_logit is None:
                    return left_logit
                else:
                    left_logit, right_logit = same_size(left_logit, right_logit)
                    return np.add(left_logit, right_logit)

            elif self.network.node[node]['type'] == NodeType.input:
                return self.network.node[node]['constant']

            elif self.network.node[node]['type'] == NodeType.kronecker:
                if left_logit is None:
                    result = np.kron(right_logit, right_logit)
                    return result
                elif right_logit is None:
                    result = np.kron(left_logit, left_logit)
                    return result
                else:
                    return np.kron(left_logit, right_logit)

            elif self.network.node[node]['type'] == NodeType.hadamard:
                if left_logit is None:
                    return right_logit
                elif right_logit is None:
                    return left_logit
                else:
                    left_logit, right_logit = same_size(left_logit, right_logit)
                    return np.multiply(left_logit, right_logit)

        except MemoryError:
            return None


# returns left and right so that they are the same size,
# assuming both are square and share some common factor
# of their sides
def same_size(left, right):

    left_row_size = np.ma.size(left, 0)
    right_row_size = np.ma.size(right, 0)
    if left_row_size != right_row_size:
        if left_row_size < right_row_size:
            length = int(right_row_size / left_row_size)
            left = np.kron(left, np.ones((length, length)))
        else:
            length = int(left_row_size / right_row_size)
            right = np.kron(right, np.ones((length, length)))

    return left, right


# Testing Genotype stuff

# innovation_db = InnovationDB()
# genome = Genome(innovation_db)
# genome2 = Genome(innovation_db)

# for i in range(0, 10):
#    genome.mutate()

    # print("printing nodes")
    # for node in genome.network.nodes(data=True):
    #    print(node)

    # print("printing edges")
    # for edge in genome.network.edges(data=True, keys=True):
    #    print(edge)

    # print("getting phenotype")
    # phenotype = genome.get_phenotype()
    # print(phenotype)

# for i in range(0, 10):
#    genome2.mutate()

# print(genome.distance(genome2))

# offspring = genome.recombinate(genome2)

# print("printing edges of genome")
# for edge in genome.network.edges(data=True, keys=True):
#    print(edge)

# print("printing edges of genome2")
# for edge in genome2.network.edges(data=True, keys=True):
#    print(edge)

# print("printing edges of offspring")
# for edge in offspring.network.edges(data=True, keys=True):
#    print(edge)

# for node in offspring.network.nodes(data=True):
#    print(node)

# print(offspring.get_phenotype())
