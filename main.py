import json
import os
import random
from datetime import datetime
import numpy
from deap import base, creator, tools
import plots
import markov
import deap_ops
import constants


def run_ga(file_in, random_seed, novelty_method):

    # set random seed
    random.seed(random_seed)
    numpy.random.seed(random_seed)

    root_out = "data/out/" + file_in + "/"
    dir_out = root_out + novelty_method + "_" + str(random_seed) + "_" + datetime.now().strftime("%Y%m%d-%H.%M.%S") + "/"

    # Create target dir if don't exist
    if not os.path.exists(root_out):
        os.mkdir(root_out)
    # Create dir_out if don't exist
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    else:
        print("Directory ", dir_out, "already exists")

    # read input model and form classes
    # for generation and evaluation of individuals
    mfi = "data/models/" + file_in
    if os.path.exists(mfi):
        tps, alphabet = markov.load_model(mfi)
    else:
        print("ERROR: no model dir")
        return 0

    # time
    start_time = datetime.now()

    # init archive
    archive = []

    # STATS
    stats = dict()
    stats["const"] = dict()
    stats["const"]["file_in"] = file_in
    stats["const"]["NGEN"] = constants.NGEN
    stats["const"]["POP_SIZE"] = constants.POP_SIZE
    stats["const"]["N_ELITE"] = constants.N_ELITE
    stats["const"]["NOV_T_MIN"] = constants.NOV_T_MIN
    stats["const"]["NOV_T_MAX"] = constants.NOV_T_MAX
    stats["const"]["NOV_FIT_THRESH"] = constants.NOV_FIT_THRESH
    stats["method"] = novelty_method

    # for plot
    fits = []
    novs = []
    arch_s = []

    # DEAP
    # toolbox
    toolbox = base.Toolbox()
    # init DEAP fitness and individual for tournament in novelty search
    if not hasattr(creator, "FitnessMaxTN"):
        creator.create("FitnessMaxTN", base.Fitness, weights=(-1.0,))
        creator.create("IndividualTN", list, fitness=creator.FitnessMaxTN)
    # init DEAP fitness and individual
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(-1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox.register("dirInd", lambda: deap_ops.create_individual(alphabet))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.dirInd)
    # GA operators
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
    # selection
    toolbox.register("select", tools.selSPEA2)
    # eval
    toolbox.register("evaluate", lambda x: (deap_ops.eval_fitness(x, tps), 0))
    toolbox.register("evaluateMulti", lambda x: deap_ops.eval_fitness_and_novelty(x, tps, pop, archive))

    # evaluation function: (fitness or fitness-novelty)
    evaluation_function = toolbox.evaluate
    feasible_individuals = 0
    # create the population
    pop = toolbox.population(n=constants.POP_SIZE)

    # generations
    for g in range(constants.NGEN):

        # new stats page
        stats[g] = dict()

        # novelty search: choose evaluate function (fitness or multi)
        if novelty_method.find("fitness_only") == -1:
            if feasible_individuals >= constants.NOV_T_MAX:
                # fitness + novelty
                evaluation_function = toolbox.evaluateMulti
            elif feasible_individuals <= constants.NOV_T_MIN:
                # fitness
                evaluation_function = toolbox.evaluate

        ###################################################################

        # EVALUATION
        # t1 = datetime.now()
        feasible_individuals = 0
        fit_values = list(map(evaluation_function, pop))
        for ind, fit in zip(pop, fit_values):
            ind.fitness.values = fit
            # count feasible individuals for novelty search
            if fit[0] > constants.NOV_FIT_THRESH:
                feasible_individuals = feasible_individuals + 1
        # print("Eval... time: ", (datetime.now() - t1).total_seconds(), "s.")

        # SELECTION
        offspring = list(map(toolbox.clone, toolbox.select(pop, k=constants.POP_SIZE - constants.N_ELITE)))
        elite = list(map(toolbox.clone, offspring[:constants.N_ELITE]))  # Select the elite

        # random.shuffle(offspring)

        # CROSSOVER
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < constants.CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # MUTATION
        for mutant in offspring:
            if random.random() < constants.MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        # t2 = datetime.now()
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        values = toolbox.map(evaluation_function, invalid_ind)
        for ind, fit in zip(invalid_ind, values):
            ind.fitness.values = fit
        # print("Eval invalid...", "time: " + str((datetime.now() - t2).total_seconds()))

        # new pop
        pop[:] = elite + offspring
        ###################################################################
        # SAVE STATISTICS

        # print used method
        # if evaluation_function == toolbox.evaluate:
        #     print(g, ":", bc.PASS + "F" + bc.ENDC, "fi=" + str(feasible_individuals), "a=" + str(len(archive)),
        #           "time: " + str((datetime.now() - t1).total_seconds()))
        # elif evaluation_function == toolbox.evaluateMulti:
        #     print(g, ":", bc.BLUE + "H" + bc.ENDC, "fi=" + str(feasible_individuals), "a=" + str(len(archive)),
        #           "time: " + str((datetime.now() - t1).total_seconds()))
        # else:
        #     print("FATAL ERROR: NO METHOD FOUND")

        res = [ind.fitness.values for ind in pop]
        fits.append(sum(x[0] for x in res) / constants.POP_SIZE)
        novs.append(sum(x[1] for x in res) / constants.POP_SIZE)
        arch_s.append(len(archive))

        # save stats
        # in case use copy.deepcopy()
        stats[g]["method"] = "F" if evaluation_function == toolbox.evaluate else "H"
        # stats[g]["method"] = "H"
        stats[g]["pop"] = pop[:]
        stats[g]["fitness"] = res[:]
        stats[g]["archive"] = archive[:]

    # end ga

    ###############################################################
    #                   OUT, PLOTS and GRAPHS
    ###############################################################
    stats["time"] = (datetime.now() - start_time).total_seconds()

    pop_plot = {"fits": [], "novs": []}
    best_plot = {"fits":[], "novs":[]}

    for pb in pop:
        pop_plot["fits"].append(pb.fitness.values[0])
        pop_plot["novs"].append(pb.fitness.values[1])

    bests = toolbox.select(pop, k=7)
    for i,bb in enumerate(bests):
        best_plot["fits"].append(bb.fitness.values[0])
        best_plot["novs"].append(bb.fitness.values[1])

    print("time elapsed :", stats["time"], "sec.")

    # save stats
    with open(dir_out + "stats.json", "w") as fp:
        json.dump(stats, fp, default=markov.serialize_sets)

    # plots.plot_fits(dir_out, constants.NGEN, fits, novs, stats["method"])
    plots.plot_data(dir_out, constants.NGEN, fits, novs, arch_s, stats["method"])
    plots.plot_pareto(dir_out, pop_plot, best_plot, stats["method"])


if __name__ == "__main__":
    run_ga("input", 8, "fitness_only")
