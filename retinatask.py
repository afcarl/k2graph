import numpy as np
import math
from population import Population
from cartographer import Cartographer
from visualize import *
from innovationdb2 import InnovationDB2
import operator
import networkx as nx
import sys
import constants
from k2graph import K2GraphGenome
import random


def is_right(full_image):
    right_objects = [0b1011, 0b0111, 0b1110, 0b1101, 0b0010, 0b0001, 0b0011, 0b1111]

    for i in range(len(right_objects)):
        image = full_image & 0b1111
        if right_objects[i] == image:
            return True

    return False


def is_left(full_image):
    left_objects = [0b1000, 0b0100, 0b1100, 0b1111, 0b1011, 0b0111, 0b1110, 0b1101]

    for i in range(len(left_objects)):
        image = (full_image & 0b11110000) >> 4
        if left_objects[i] == image:
            return True

    return False

def convert_to_image(binary):
    image = np.zeros((10, 1))

    for i in range(8):
        if binary & 1 == 1:
            image[7 - i] = 3
        else:
            image[7 - i] = -3

        binary = binary >> 1

    image[8] = 3
    image[9] = 3

    return image

# Setting up the dictionary of inputs with the labeled outputs
label_mat = np.zeros((2,256))
image_mat = np.zeros((10,256))

for i in range(256):

    left = is_left(i)
    right = is_right(i)

    if left and right:
        label_mat[:,i] = [1, 1]
    elif left:
        label_mat[:,i] = [1, 0]
    elif right:
        label_mat[:,i] = [0, 1]
    else:
        label_mat[:,i] = [0, 0]

    image_mat[:,i] = convert_to_image(i).flatten()


def activation(x):
    #print(np.shape(x))
    return np.maximum(x, 0.01*x, x)
    #y = [x >= 0]
    #yy = np.logical_not(y)

    #z = 0.5 * np.expm1(yy * x) + (y * x)

    #print(np.shape(z))

def function(g):

    if nx.dag_longest_path_length(g.network) > 2:
        print("skip")
        return 0

    pheno = g.get_phenotype()

    nodes = cart.get_nodes(int(math.sqrt(np.size(pheno))), two_dimensional, symmetric, input_nodes=True)
    output_nodes = cart.get_nodes(int(math.sqrt(np.size(pheno))), two_dimensional, symmetric, input_nodes=False)

    total_error = 0

    #for i in range(256):
    # set input values and activate network

    result = np.zeros((int(math.sqrt(np.size(pheno))), 256))

    input_vector = np.zeros((int(math.sqrt(np.size(pheno))), 256))

    for j in range(len(nodes)):
        input_vector[nodes[j]] += image_mat[j]


    # depth is 4
    for j in range(4):
        result = np.matmul(np.transpose(pheno), input_vector + result)
        result = activation(result)

    output_vector = np.zeros((2, 256))

    for j in range(len(output_nodes)):
        output_vector[j,:] += result[output_nodes[j]]

    output_vector = np.clip(output_vector, 0, 1)

    # calculate the error for this sample
    total_error = np.sum(np.abs(output_vector - label_mat))

    return 1000.0 / (1.0 + total_error ** 2)


if __name__ == "__main__":
    # cartesian locations of inputs and bias
    inputs = [(-1, -1, 1),
             (-0.33333, -1, 1),
             (-1, -1, -1),
             (-0.3333, -1, -1),
             (0.33333, -1, 1),
             (1, -1, 1),
             (0.3333, -1, -1), 
             (1, -1,  -1),
             (-1, -1, 0), # (0, -1, 0)
             (1, -1, 0)]

    # cartesian location of outputs
    outputs = [(-1, 1, 0), (1, 1, 0)]
    cart = Cartographer(inputs, outputs)
    two_dimensional = False
    symmetric = True

    for i in range(40):

        K2GraphGenome.P_ADD_NODE = 0.1
        K2GraphGenome.P_ADD_LINK = 0.6
        K2GraphGenome.P_NODE = 0.5#random.uniform(0.2, 0.8)
        K2GraphGenome.P_LINK = 0.5#random.uniform(0.2, 0.8)
        K2GraphGenome.MUTATE_STD = 0.25
        constants.SURVIVAL_THRESHOLD = 0.8

        # test each random hyperparameter set twice
        for j in range(1):

            innovationdb = InnovationDB2()
            seed_genome = K2GraphGenome(innovationdb)
            pop = Population(seed_genome)
            pop.run(function, 2000, i, j)

            pheno = pop.best_genome.get_phenotype()

            #input_nodes = cart.get_nodes(int(math.sqrt(np.size(pheno))), two_dimensional, symmetric, input_nodes=True)
            #output_nodes = cart.get_nodes(int(math.sqrt(np.size(pheno))), two_dimensional, symmetric, input_nodes=False)

            # Warshall's algorithm computes the transitive closure of an adjacency matrix
            # WARNING: This function is very computationally expensive. That's why I only use it for visualization
            def warshall(a):
                n = a.shape[0]
                for k in range(n):
                    for i in range(n):
                        for j in range(n):
                            a[i][j] = a[i][j] or (a[i][k] and a[k][j])
                return a

            # Removes links to and from unreachable nodes in the adjacency matrix
            def remove_unreachable_node_links(x, input_nodes, output_nodes):

                n = x.shape[0]
                # prep adjacency matrix for Warshall's algorithm
                prep = x != 0.0
                prep += np.eye(n, dtype=bool)

                # get the transitive closure for the prepared matrix
                transitive = warshall(prep)
                print("finished warshall")

                # find all nodes that are reachable from the input nodes and reach the output nodes.
                reachable_nodes = np.zeros((1, n), dtype=bool)
                for input_node in input_nodes:
                    for output_node in output_nodes:
                        reachable_nodes |= transitive[input_node] & transitive[:,output_node]

                # clear the rows and columns of the unreachable nodes
                for i in range(n):
                    if not reachable_nodes[0][i]:
                        x[i] = np.zeros(n)
                        x[:,i] = np.zeros(n)

                return x

    #pheno = np.array(pheno)
    #pheno = remove_unreachable_node_links(pheno, input_nodes, output_nodes)

    #input_vector = np.zeros((int(math.sqrt(np.size(pheno))), 1))

    #for j in range(len(nodes)):
    #    input_vector[nodes[j]] += convert_to_image(32)[j]
    '''

    result = np.zeros((int(math.sqrt(np.size(pheno))), 1))

    for j in range(4):
        result = np.matmul(np.transpose(pheno), input_vector + result)
        result = activation(result)
        print(j)
        print(result)

    '''
    # visualize the genome
    #visualize(pheno, two_dimensional, symmetric)
