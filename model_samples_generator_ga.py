# library imports
import time
import random
import numpy as np

# project import
from settings import *
from vector import Vector


class ModelSamplesGeneratorGeneticAlgorithm:
    """
    This class responsible to find the best option for action given an agent's state using the GA algorithm
    """

    # CONSTS #
    GENE_POP_SIZE = 100
    GENERATIONS = 50
    MUTATION_RATE = 0.05
    MUTATION_INF = 5
    ROYALTY_RATE = 0.05
    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run(start_guess,
            my_vel,
            neighbors):
        # init population
        population = [start_guess for _ in range(ModelSamplesGeneratorGeneticAlgorithm.GENE_POP_SIZE)]
        best_gene = population[0]
        for generation in range(ModelSamplesGeneratorGeneticAlgorithm.GENERATIONS):
            ModelSamplesGeneratorGeneticAlgorithm.mutation(population=population)
            fitness = ModelSamplesGeneratorGeneticAlgorithm.fitness(population=population,
                                                                    neighbors=neighbors,
                                                                    my_vel=my_vel)
            best_gene = population[np.argmax(fitness)]
            population = ModelSamplesGeneratorGeneticAlgorithm.selection(population=population,
                                                                         fitness=fitness)
        return best_gene

    @staticmethod
    def selection(population,
                  fitness):
        # convert scores to probability
        max_fitness = max(fitness)
        reverse_scores = [max_fitness - score for score in fitness]
        sum_fitness = sum(reverse_scores)
        if sum_fitness > 0:
            fitness_probabilities = [score / sum_fitness for score in reverse_scores]
        else:
            fitness_probabilities = reverse_scores
        # sort the population by fitness
        genes_with_fitness = zip(fitness_probabilities, population)
        genes_with_fitness = sorted(genes_with_fitness, key=lambda x: x[0], reverse=True)
        # pick the best royalty_rate anyway
        royalty = [val[1] for val in genes_with_fitness[:round(len(genes_with_fitness)*ModelSamplesGeneratorGeneticAlgorithm.ROYALTY_RATE)]]
        # tournament around the other genes
        left_genes = [val[1] for val in genes_with_fitness[round(len(genes_with_fitness) * ModelSamplesGeneratorGeneticAlgorithm.ROYALTY_RATE):]]
        left_fitness = [val[0] for val in genes_with_fitness[round(len(genes_with_fitness) * ModelSamplesGeneratorGeneticAlgorithm.ROYALTY_RATE):]]
        pick_genes = []
        left_count = len(population) - len(royalty)
        while len(pick_genes) < left_count:
            pick_gene = random.choices(left_genes, weights=left_fitness)
            pick_genes.append(pick_gene)
        # add the royalty
        pick_genes = list(pick_genes)
        pick_genes.extend(royalty)
        return pick_genes

    @staticmethod
    def mutation(population):
        new_population = []
        for gene in population:
            if random.random() < ModelSamplesGeneratorGeneticAlgorithm.MUTATION_RATE:
                new_population.append(gene.add(Vector.random().mult(scalar=random.random()*ModelSamplesGeneratorGeneticAlgorithm.MUTATION_INF)))
            else:
                new_population.append(gene)
        return new_population

    @staticmethod
    def fitness(population,
                my_vel,
                neighbors):
        # neighbor size declare
        NEIGHBOR_SIZE = 4
        # split to a list of each neighbor
        each_neighbor = [tuple(neighbors[start_index:start_index + NEIGHBOR_SIZE]) for start_index in range(0, len(neighbors), NEIGHBOR_SIZE)]
        if not each_neighbor:
            avg_vel = my_vel
        else:
            sum_vel_x, sum_vel_y = 0, 0
            for neigh in each_neighbor:
                sum_vel_x += neigh[2]
                sum_vel_y += neigh[3]
            # avg and transform vel to relative velocity
            avg_vel = Vector(sum_vel_x / len(each_neighbor), sum_vel_y / len(each_neighbor)).sub(my_vel)
        return [1 - abs(avg_vel.angle_between(gene)) for gene in population]
