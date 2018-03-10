import copy
from innovationdb2 import InnovationDB2
import networkx as nx
import numpy as np
import random
from visualize import *
import constants

# Written by Stefano Palmieri in February 2017

# This class stores the genome as a weighted Multi-Directed Acyclical
# Graph called an K^2 Graph. The K^2 Graph is based on K^2 Trees
# (see: Nieves R. Brisaboa , Susana Ladra , Gonzalo Navarro, k2-Trees
# for Compact Web Graph Representation, Proceedings of the 16th International
# Symposium on String Processing and Information Retrieval, 
# August 25-27, 2009, Saariselkä, Finland). The difference is that nodes can 
# have multiple incoming links in a K^2 Graph. Hence there is a directed 
# acyclical graph structure instead of a tree structure.

class K2GraphGenome(object):
    # Probability of adding a new node
    P_ADD_NODE = constants.P_ADD_NODE # this was 0.25, changed to 0.1

    # Probability of adding a new link
    P_ADD_LINK = constants.P_ADD_LINK   

    # Probability of mutating existing node
    P_NODE = constants.P_NODE

    # Probability of mutating existing link
    P_LINK = constants.P_LINK

    #These need to be implemented, being careful to avoid cycles
    #when nodes or links are deleted
    # Probability of deleting existing node
    P_DEL_NODE = 0.0

    # Probability of deleting existing link
    P_DEL_LINK = 0.0

    # Probability of zeroing existing node weight
    P_DEL_NODE_WEIGHT = 0.01
    
    # Sizes for matrix axes in the node constants
    AXIS_SIZE = 8

    # Mean and standard deviation for weight mutations
    MUTATE_MEAN = 0.00
    MUTATE_STD = constants.MUTATE_STD

    # Constants and Coefficients for computing compatibility
    EXCESS_COEFFICIENT = 1.0
    DISJOINT_COEFFICIENT = 2.0
    WEIGHT_COEFFICIENT = 0.4

    def __init__(self, innovation_db):
        self.innovation_db = innovation_db
        self.network = nx.MultiDiGraph()

        self.network = nx.MultiDiGraph()
        self.network.add_node(1, constant=np.zeros((self.AXIS_SIZE, 
                                                   self.AXIS_SIZE)))

        self.innovation_db.retrieve_node_innovation_num(0, 0)
        self.phenotype_dict = {}

        self.fitness = 1
        self.adjusted_fitness = 1

    def mutate(self):

        # If mutation occurs, mutate an existing node
        if random.random() <= self.P_NODE:
            #print("1")
            self.mutate_node()

        # If mutation occurs, mutate an existing link
        if random.random() <= self.P_LINK:
            #print("2")
            self.mutate_link()

        # If mutation occurs, add a new node
        if random.random() <= self.P_ADD_NODE:
            #print("3")
            self.mutate_add_node()

        # If mutation occurs, delete a node
        if random.random() <= self.P_DEL_NODE:
            #print("6")
            self.mutate_del_node()

        # If mutation occurs, add a new link
        if random.random() <= self.P_ADD_LINK:
            #print("4")
            self.mutate_add_link()

         # If mutation occurs, delete a link
        if random.random() <= self.P_DEL_LINK:
            #print("5")
            self.mutate_del_link()

        if random.random() <= self.P_DEL_NODE_WEIGHT:
            self.mutate_del_node_weight()

    def random_weight(self):
        return np.random.normal(self.MUTATE_MEAN, self.MUTATE_STD)

    # Helper method that returns a non-empty node and a non-empty key
    def choose_node_and_key(self):
        # select a random node to start a link from, 
        # where the node's cell is not zero.
        nodes = [node for node, data
                           in self.network.nodes(data=True)
                           if np.any(data['constant'])]
        if not nodes:
            return None, None

        node = random.choice(nodes)

        key_found = False

        while not key_found:
            key = random.randint(0, self.AXIS_SIZE * self.AXIS_SIZE - 1)
            if self.network.node[node]['constant'].item(key) != 0:
                return node, key

    # Create a new node by selecting an existing node and spawning
    # a child from it
    def mutate_add_node(self):

        # choose a parent and corresponding key
        #parent, key = self.choose_node_and_key() just changed this sunday feb 26 to the next to lines
        
        parent = random.choice(self.network.nodes())
        key = random.randint(0, self.AXIS_SIZE * self.AXIS_SIZE - 1)
        
        if parent is None:
            return

        child = self.innovation_db.retrieve_node_innovation_num(key, parent)

        '''
        self.network.add_node(child,
                      constant=np.zeros((self.AXIS_SIZE,
                                         self.AXIS_SIZE),
                                        dtype=np.float16))
        '''
        '''
        self.network.add_node(child,
              constant=np.ones((self.AXIS_SIZE,
                               self.AXIS_SIZE)))
        '''
        if random.random() < 0.9:   
            self.network.add_node(child,
                constant=copy.deepcopy(self.network.node[parent]['constant']))
        else:
            self.network.add_node(child,
                constant=np.zeros((self.AXIS_SIZE, self.AXIS_SIZE), dtype=np.float16))

        
        innovation = self.innovation_db.retrieve_link_innovation_num(key,
                                                                     parent, 
                                                                     child)


        self.network.add_edge(parent, child, key=key,
                              weight=1.0, # this is correct
                              innovation=innovation)

    # Delete an existing node
    def mutate_del_node(self):

        # choose a node to delete
        nodes = [node for node in self.network.nodes() if node != 1]

        if nodes:
            node = random.choice(nodes)

            successors = self.network.successors(node)

            print("'cessors of ", node)
            print(successors)
            print("end")


    # Create a new link and add it to the Genome.
    def mutate_add_link(self):

        # choose a parent and corresponding key
        parent, key = self.choose_node_and_key()
        
        #parent = random.choice(self.network.nodes())
        #key = random.randint(0, self.AXIS_SIZE * self.AXIS_SIZE - 1)
        
        if parent is None:
            return

        child = random.choice(self.network.nodes())

        innovation = self.innovation_db.retrieve_link_innovation_num(key, 
                                                                     parent,
                                                                     child)

        self.network.add_edge(parent, child, key=key,
                              weight=self.random_weight(),
                              innovation=innovation)

        # Check for cycles. If there are, remove this link
        cycles = nx.has_path(self.network,parent,child) and \
                 nx.has_path(self.network,child,parent)

        if cycles:
            self.network.remove_edge(parent, child, key=key)

    # Create a new link and add it to the Genome.
    def mutate_del_link(self):

        # choose a link
        edges = self.network.edges(keys=True)

        if edges:
            # Select a random link
            start, end, key = random.choice(self.network.edges(keys=True))

            self.network.remove_edge(start, end, key=key)


    # Randomly mutates one of the existing nodes.
    def mutate_node(self):

        # get list of nodes
        nodes = self.network.nodes()

        # randomly pick a node from the list
        node = random.choice(nodes)

        # mutate one of it's constant's elements
        rand_row = random.randrange(0, self.AXIS_SIZE)
        rand_col = random.randrange(0, self.AXIS_SIZE)

        # Mutate by Gaussian random variable
        change = self.random_weight()

        # apply the change
        self.network.node[node]['constant'][rand_row, rand_col] += change

    # Randomly mutate an existing link by modifying the weight
    def mutate_link(self):

        edges = self.network.edges(keys=True)

        if edges:
            # Select a random link
            start, end, key = random.choice(self.network.edges(keys=True))

            # Add a Gaussian random variable to existing weight
            weight_change = self.random_weight()

            # Modify the weight on the link
            self.network[start][end][key]['weight'] += weight_change

    def mutate_del_node_weight(self):

        # get a node and a respective key
        node, key = self.choose_node_and_key()

        if node:

            # clear the weight
            self.network.node[node]['constant'].itemset(key, 0) 


    # Returns a recombinated genome taken between this organism and another
    def recombinate(self, other):

        # Identify dominant and recessive parents
        if self.fitness >= other.fitness:
            dominant = self
            recessive = other
        else:
            dominant = other
            recessive = self

        offspring = K2GraphGenome(self.innovation_db)
        offspring.inherit_genes(dominant, recessive)

        if not nx.is_directed_acyclic_graph(offspring.network):
            print("Something went wrong, not a DAG")

        #offspring.remove_cycles()

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

    def inherit_link_genes(self, dominant, recessive):
         # Inherit link genes
        for edge in dominant.network.edges_iter(data=True, keys=True):
            s, e, key, data = edge
            matched = False
            for edge2 in recessive.network.edges_iter(data=True, keys=True):
                d, r, key2, data2 = edge2
                if data['innovation'] == data2['innovation']:
                    matched = True
                    # randomly inherit homologous genes
                    if random.random() < 0.5:
                        self.network.add_edge(s, e,
                                              key=key,
                                              weight=data['weight'],
                                              innovation=data['innovation'])
                    else:
                        self.network.add_edge(d, r,
                                              key=key2,
                                              weight=data2['weight'],
                                              innovation=data2['innovation'])
                    break
            # if there wasn't a match, this is a disjoint or excess gene
            if not matched:
                self.network.add_edge(s, e,
                                       key=key,
                                       weight=data['weight'],
                                       innovation=data['innovation'])

    # Have this genome inherit genes from its parents
    def inherit_genes(self, dominant, recessive):

        self.inherit_link_genes(dominant, recessive)

        # Inherit node genes
        for node, data in dominant.network.nodes_iter(data=True):
            matched = False
            for node2, data2 in recessive.network.nodes_iter(data=True):
                if node == node2:
                    matched = True
                    # randomly inherit homologous genes
                    if random.random() < 0.5:
                        self.network.add_node(node, copy.deepcopy(data))
                    else:
                        self.network.add_node(node2, copy.deepcopy(data2))
                    break
            # if there wasn't a match, this is a disjoint or excess gene
            if not matched:
                self.network.add_node(node, copy.deepcopy(data))

    def get_link_innovations(self):
        innovations = [innovation for s, e, innovation
                       in self.network.edges_iter(data='innovation')]
        return innovations

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

    def get_node_innovations(self):

        innovations = [innovation for innovation in self.network.nodes_iter()]
        return innovations

    def node_distance(self, other):
        innovations = self.get_node_innovations()
        other_innovations = other.get_node_innovations()

        innovations_set = set(innovations)
        other_innovations_set = set(other_innovations)

        intersection = innovations_set & other_innovations_set

        total_diff = 0

        for i in intersection:
            abs_diff = np.fabs(self.network.node[i]['constant'] \
                       - other.network.node[i]['constant'])

            total_diff += np.sum(abs_diff)
        
        return total_diff

        # Compute the distance using innovation numbers on the link genes
    def distance(self, other):
        innovations = self.get_link_innovations()
        other_innovations = other.get_link_innovations()

        if not innovations or not other_innovations:
            return 0

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

    # returns the max shape of the sub-phenotypes of nodes in the list
    def get_max_shape(self, nodes):
        shape = lambda s: self.phenotype_dict[s].shape
        return max(shape(s) for s in nodes)

    # This method prepares a matrix to be added to the phenotype matrix
    def get_matrix(self, node, end, key, data, max_shape):
        cell_weight = self.network.node[node]['constant'].item(key)

        temp = np.multiply(cell_weight * data['weight'],
                           self.phenotype_dict[end])

        if temp.shape != max_shape:
            temp = np.kron(temp,
                           np.ones((int(max_shape[0] / temp.shape[0]), 
                                    int(max_shape[1] / temp.shape[1]))
                           ))
        return temp

    def compose_subphenotype(self, node, successors):

        max_shape = self.get_max_shape(successors)

        try:
            sub = np.kron(self.network.node[node]['constant'],
                          np.ones(max_shape))

            key_set = set()

            for edge in self.network.out_edges(node, keys=True, data=True):
                start, end, key, data = edge
                left = int((key / self.AXIS_SIZE)) * max_shape[0]
                right = left + max_shape[0]
                top = int((key % self.AXIS_SIZE)) * max_shape[1]
                bottom = top + max_shape[1]

                # clear the phenotype area once so that mutiple subnodes add
                if key not in key_set:
                    sub[left:right,top:bottom] = np.zeros(max_shape)

                key_set.add(key)

                sub[left:right,top:bottom] += self.get_matrix(*edge, max_shape)
        except MemoryError:
            return self.network.node[node]['constant']

        return sub

    def get_subphenotype(self, node):
        successors = self.network.successors(node)

        if successors:
            return self.compose_subphenotype(node, successors)
        # leaf node
        else:    
            return self.network.node[node]['constant']

    # Compose phenotype from subphenotypes using postorder, starting from the 
    # root node
    def get_phenotype(self):
        for node in nx.dfs_postorder_nodes(self.network,1):
            self.phenotype_dict[node] = self.get_subphenotype(node)

        temp = self.phenotype_dict[1]
        self.phenotype_dict.clear()
        return temp


innovation_db = InnovationDB2()
genome = K2GraphGenome(innovation_db)
genome2 = K2GraphGenome(innovation_db)

for i in range(20):
    genome.mutate()
    genome2.mutate()

print("printing nodes")
for node in genome.network.nodes(data=True):
    print(node)

print("printing nodes")
for node in genome2.network.nodes(data=True):
    print(node)

print(genome.distance(genome2))


'''
genome2 = K2GraphGenome(innovation_db)

for i in range(60):
    genome.mutate()
    genome2.mutate()
print("printing nodes")
for node in genome.network.nodes(data=True):
    print(node)
print("printing edges")
for edge in genome.network.edges(data=True, keys=True):
    print(edge)
print("printing nodes")
for node in genome2.network.nodes(data=True):
    print(node)
print("printing edges")
for edge in genome2.network.edges(data=True, keys=True):
    print(edge)
    
genome3 = genome.recombinate(genome2)
print(genome3.get_phenotype())
print("printing nodes")
for node in genome3.network.nodes(data=True):
    print(node)
print("printing edges")
for edge in genome3.network.edges(data=True, keys=True):
    print(edge)
# choose a node to delete
nodes = [node for node in genome3.network.nodes() if node != 1]
if nodes:
    node = random.choice(nodes)
    cessors = genome3.network.successors(node)
    print("'cessors of ", node)
    print(cessors)
    print("end")
    for edge in genome3.network.out_edges(node, keys=True):
        start, end, key = edge
        if start == node and end in cessors:
    genome3.network.remove_node(node)
    print("printing nodes")
    for node in genome3.network.nodes(data=True):
        print(node)
    print("printing edges")
    for edge in genome3.network.edges(data=True, keys=True):
        print(edge)
        '''