import copy
#from constants import *
from multiprocessing import Pool
from species import SpeciesSet
import constants
import networkx as nx
import csv
from k2graph import K2GraphGenome

#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# If it is not possible or desirable to put the notice in a particular
# file, then You may include the notice in a location (such as a LICENSE
# file in a relevant directory) where a recipient would be likely to look
# for such a notice.

# You may add additional accurate notices of copyright ownership.

# Written by Stefano Palmieri in January 2017

class Population(object):

    def __init__(self, seed_genome):

        self.best_genome = None
        self.best_species = None
        self.species_set = SpeciesSet()
        self.generation = -1

        self.species_set.speciate([seed_genome])

        # copy the seed genome for generation 0
        offspring = self.reproduce(0)
        self.species_set.speciate(offspring)

    def reproduce(self, num_elites):

        # calculate the allowed number of offspring
        # and tell the species to reproduce.

        # The reason for doing things this way is to avoid
        # a gap where the full population size isn't used.
        # (See "The Extra Pixel Problem" or "The Thin White Stripe"

        offspring = []

        population_fitness = 0

        for s in self.species_set.species:
            population_fitness += s.total_adjusted_fitness()

        running_fraction = 0

        # Each species reproduces in proportion to the
        # total adjusted fitness of that species
        for s in self.species_set.species:
            top = int((constants.POPULATION_SIZE - num_elites) * running_fraction)
            running_fraction += s.total_adjusted_fitness() / population_fitness
            next_top = int((constants.POPULATION_SIZE - num_elites) * running_fraction)
            offspring.extend(s.reproduce(next_top - top))

        return offspring

    def run(self, fitness_function, generations, iter, train):

        filename = str(iter) + "-" + str(train) + ".csv"
        with open(filename, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

            for generation in range(generations):
                self.generation += 1

                self.calculate_fitnesses(fitness_function)

                self.report()

                if self.best_genome.fitness > constants.MAX_FITNESS:
                    break

                spamwriter.writerow([self.best_genome.fitness])

                # Kill off inferior organisms
                self.species_set.cull()

                elites = self.elites()

                # Create the next generation from the current generation.
                offspring = self.reproduce(len(elites))

                # Add in elites
                offspring.extend(elites)

                # Clear out the old population
                self.species_set.clear_species()

                # Update species age.
                for s in self.species_set.species:
                    s.age += 1

                # Divide the new population into species.
                self.species_set.speciate(offspring)
                
            spamwriter.writerow([constants.SURVIVAL_THRESHOLD, K2GraphGenome.P_ADD_NODE, K2GraphGenome.P_ADD_LINK, K2GraphGenome.P_NODE, K2GraphGenome.P_LINK, K2GraphGenome.MUTATE_STD ])

    def calculate_fitnesses(self, fitness_function):

        lowest = None
        highest = None
        hg = None
        hs = None


        
        for s in self.species_set.species:
            pool = Pool(24) 

            results = pool.map(fitness_function, s.members)

            pool.close()
            pool.join()

            count = 0

            for o in s.members:

                fitness = results[count]
                o.fitness = fitness

                if highest is None or fitness > highest:
                    highest = fitness
                    hg = o
                    hs = s

                if lowest is None or fitness < lowest:
                    lowest = fitness

                count += 1


        # the reason for adding lowest fitness to o.fitness is to handle
        # the case when fitnesses are negative
        for s in self.species_set.species:
            for o in s.members:
                o.adjusted_fitness = (o.fitness + abs(lowest)) / len(s.members)

        if self.best_genome is None or hg.fitness >= self.best_genome.fitness:
            self.best_genome = hg
            self.best_species = hs

    def elites(self):

        elites = []

        for s in self.species_set.species:
            elites.extend(s.members[0 : constants.ELITISM: 1])

        return elites

    def report(self):

        print("___GENERATION", self.generation, "___")
        print("Number of species: ", len(self.species_set.species))
        for specie in self.species_set.species:
            print("species id:", specie.id_, " size: ", len(specie.members))
        print("  Best Organism: ")
        print("    species ", self.best_species.id_)
        print("    fitness", self.best_genome.fitness)
        print("    adjusted fitness", self.best_genome.adjusted_fitness)

        #two_dimensional = False
        #symmetric = True

        #visualize(self.best_genome.get_phenotype(), two_dimensional, symmetric)

        '''
        if self.generation % 100 == 0:
            pheno = self.best_genome.get_phenotype()
            pheno = imresize(pheno, (32, 32), interp='nearest')
            name = str(self.generation) + '.png'
            imsave( name, pheno)

        '''
        
    
        #print("printing nodes")
        #for node in self.best_genome.network.nodes(data=True):
        #    print(node)

        #print("printing edges")
        #for edge in self.best_genome.network.edges(data=True, keys=True):
        #    print(edge)

        print(constants.SURVIVAL_THRESHOLD)
        print(K2GraphGenome.P_ADD_NODE)
        print(K2GraphGenome.P_ADD_LINK)
        print(K2GraphGenome.P_NODE)
        print(K2GraphGenome.P_LINK)
        print(K2GraphGenome.MUTATE_STD)

        #print("printing phenotype")
        #print(self.best_genome.get_phenotype())

        # for s in self.species_set.species:
        #    for o in s.members:
        #        print("species id: ", s.ID, ", fitness: ", o.fitness)


