import random
import math
from constants import *
import networkx as nx

class Species(object):

    def __init__(self, representative, id_):
        self.representative = representative
        self.id_ = id_
        self.age = 0
        self.members = []

        self.members.append(representative)

    def total_adjusted_fitness(self):
        hey =  sum(genome.adjusted_fitness for genome in self.members)
        return hey

    def total_fitness(self):
        return sum(genome.fitness for genome in self.members)

    # Returns the offspring of this species such that
    # the number of offspring returned is the same as
    # "size"
    def reproduce(self, size):

        offspring = []

        species_fitness = self.total_fitness()

        running_fraction = 0.0

        # Each organism reproduces in proportion to the
        # adjusted fitness of that organism
        # This part isn't working perfectly
        for m in self.members:
            top = int(round(size * running_fraction))
            running_fraction += m.fitness / species_fitness

            next_top = int(round(size * running_fraction))
            # print("organism with fitness ", m.fitness, " made ", next_top - top)
            for i in range(next_top - top):
                mate = random.choice(self.members)
                baby = m.recombinate(mate)

                #print("Something went wrong, not a DAG1")
                baby.mutate()        
                #if not nx.is_directed_acyclic_graph(baby.network):
                #    print("Something went wrong, not a DAG2222222222222222222222222")
                #    print("printing nodes")
                #    for node in baby.network.nodes(data=True):
                #        print(node)

                #    print("printing edges")
                #    for start, end, key in baby.network.edges(keys=True):
                #        print("start ", start, " end ", end, " key ", key)
                offspring.append(baby)

        return offspring

    def cull(self):
        self.members.sort(key=lambda m: m.adjusted_fitness, reverse=True)

        self.members = self.members[0 : math.ceil(SURVIVAL_THRESHOLD * len(self.members))]

class SpeciesSet(object):

    def __init__(self):
        self.index = 0
        self.species = []

    # Clear all the members
    def clear_species(self):
        for s in self.species:
            s.members.clear()

    def cull(self):
        for s in self.species:
            s.cull()

    # Find the species with the most similar representative for
    # each individual in the population and place that individual into
    # a species
    def speciate(self, population):
        for individual in population:

            min_distance = None
            closest_species = None

            for s in self.species:
                distance = individual.distance(s.representative)
                if distance < COMPATIBILITY_THRESHOLD \
                        and (min_distance is None or distance < min_distance):
                    closest_species = s
                    min_distance = distance

            if closest_species is not None:
                closest_species.members.append(individual)
            # The individual doesn't fit in an existing species, so create
            # a new one.
            else:
                self.index += 1
                self.species.append(Species(individual, self.index))

        # Only keep non-empty species.
        self.species = [s for s in self.species if s.members]

        self.select_species_representatives()

    # Select a random current member as the new representative
    # for all the species in the set
    def select_species_representatives(self):
        for s in self.species:
            s.representative = random.choice(s.members)
