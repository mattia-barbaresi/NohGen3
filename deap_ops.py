import random

import constants
import numpy as np
import novelty_search
import markov


# fun for creating an individual
def create_individual(alphabet):
    # using dirichlet distribution
    v = random.choices(alphabet, k=constants.IND_SIZE)
    return v


# fun for evaluating individuals
def eval_fitness(individual, tps):
    """
    Generates NUM_SEQS sequences with given tps-model and evaluate each sequences with given classes and patterns.
    Return percentage of hits.
    """
    # use similarity instead of perfect match
    # res = fc.evaluate_sequences2(sequences, classes["fc"], patterns)
    res = markov.sequences_markov_support_log(individual, tps)

    return res


# on genotype
def eval_fitness_and_novelty(individual, tps, population, archive):
    fit = eval_fitness(individual, tps)
    novelty_search.archive_assessment(individual, fit, archive)
    nov = novelty_search.novelty(individual, population, archive)
    return fit, nov


# decorator to normalize individuals
def normalize_individuals():
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                mn = min(child)
                if mn < 0:
                    child[:] = [i - mn for i in child]
                sm = sum(child)
                for i in range(len(child)):
                    child[i] = child[i]/sm
            return offspring
        return wrapper
    return decorator
