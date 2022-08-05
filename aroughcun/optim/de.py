"""
Basic differential evolution implementation for RACCOON
F. Comitani     @2019-2022

Based on
https://nathanrooy.github.io/posts/2017-08-27/simple-differential-evolution-with-python

Storn, R.; Price, K. (1997). "Differential evolution - a simple and efficient heuristic
for global optimization over continuous spaces". Journal of Global Optimization.
11 (4): 341–359. doi:10.1023/A:1008202821328.
"""

import random
import logging
from decimal import Decimal, ROUND_HALF_UP
from math import nan

def _clamp(x, min_val, max_val):
    """ Force a number between bounds.

    Args:
        x (float): input value to be clamped.
        min_val (float): lower bound.
        max_val (float): upper bound.

    Returns:
        (float): clamped value.
    """

    return max([min([max_val, x]), min_val])


def _tostring(x):
    """ Conbine a list of numbers into a single string with underscore as separator.

    Args:
        x (list of floats): list of numbers to combine.

    Returns:
        (str): combined string.
    """

    string = ''
    for xx in x:
        string = string + str(xx) + '_'
    return string[:-1]


def _differential_evolution(loss_fun, bounds, integers=None,
        n_candidates=10, mutation=0.6, recombination=0.7,
        maxiter=20, tol=1e-4, seed=None):
    """ Basic Differential Evolution implementation.

    Args:
        loss_fun (function): objective function, takes a set of parameters to be optimized
            and returns a single float value.
        bounds (tuple): minimum and maximum boundaries for the parameters to optimize.
        integers (list of booleans or None): list with information on which parameters
            are integers, if None (default) treat every parameter as float.
        n_candidates (int): size of the candidate solutions population.
        mutation (float): scaling factor for the mutation step.
        recombination (float): recombination (crossover) rate.
        maxiter (float): maximum number of generations.
        tol (float): solution improvement tolerance,
            if after 3 generations the best solution is not improved by at least
            this value, stop the iteration.
        seed (int): seed for the random numbers generator.

    Returns:
        (tuple (list of floats, list of objects, list of floats)): tuple containing
            the list of best parameters; 
            a list containing score, labels, clustering parameter,
            projected points, trained maps, filtered features and
            trained low-information filter from the best scoring model;
            a matrix containing all the explored models' parameters
            and their scores (useful for plotting the hyperspace).
    """

    if seed is not None:
        random.seed(seed)

    """ Randomly initialize candidated solutions. """

    allscores = {}

    # TO DO: Fix redundancy in saving the parameters
    # allparams={}

    population = []
    for i in range(n_candidates):
        indv = []
        for j in range(len(bounds)):
            if integers is not None and integers[j]:
                indv.append(random.randint(bounds[j][0], bounds[j][1]))
            else:
                indv.append(random.uniform(bounds[j][0], bounds[j][1]))

        population.append(indv)

    """ Run epochs. """

    best_res = [nan] * 7
    best_score = 1e100
    best_history = []

    for i in range(maxiter):

        logging.debug('DE generation: {:d}'.format(i + 1))

        best_history.append(best_score)

        scores_gen = []

        """ Cycle through the whole population. """

        for j in range(n_candidates):

            """ Create hybrid. """

            candid_ix = list(range(n_candidates))
            candid_ix.remove(j)
            candid_ix = random.sample(candid_ix, 3)

            agents = [population[candid_ix[0]],
                      population[candid_ix[1]],
                      population[candid_ix[2]]]

            target = population[j]

            hybrid = [x - y for x, y in zip(agents[1], agents[2])]
            hybrid = [x + mutation * y for x, y in zip(agents[0], hybrid)]
            hybrid = [_clamp(hybrid[k], bounds[k][0], bounds[k][1])
                    for k in range(len(hybrid))]

            """ Make sure integer parameters stay as such. """

            for k in range(len(bounds)):
                if integers is not None and integers[k]:
                    hybrid[k] = int(Decimal(str(hybrid[k])).to_integral_value(
                        rounding=ROUND_HALF_UP))

            """ Recombine target solution. """

            mutant = []
            for k in range(len(target)):
                if random.random() <= recombination:
                    mutant.append(hybrid[k])
                else:
                    mutant.append(target[k])

            """ Select the solution with best score,
            if candidate's score has been evaluated before, skip and use old score. 
            """

            # Note: keep the results for the best candidate so we don't have to
            # recalculate them.

            if _tostring(target) not in allscores:
                score_target, labs, eps_opt, pj, mapping,\
                    keepfeat, decomp = loss_fun(target)
                allscores[_tostring(target)] = score_target
                #allparams[_tostring(target)]=[labs, eps_opt, pj, keepfeat, decomp, parmvals]
            else:
                logging.debug('Candidate found in store')
                score_target = allscores[_tostring(target)]
                labs = None

            if _tostring(mutant) not in allscores:
                score_mutant, labs_m, eps_opt_m, pj_m, mapping_m,\
                    keepfeat_m, decomp_m = loss_fun(mutant)
                allscores[_tostring(mutant)] = score_mutant
                #allparams[_tostring(mutant)]=[labs_m, eps_opt_m, pj_m, keepfeat_m, decomp_m, parmvals_m]
            else:
                logging.debug('Candidate found in store')
                score_mutant = allscores[_tostring(mutant)]
                labs_m = None

            if score_mutant < score_target:
                population[j] = mutant
                labs, eps_opt, pj, mapping, keepfeat, decomp = labs_m, eps_opt_m, pj_m,\
                    mapping_m, keepfeat_m, decomp_m
                scores_gen.append(score_mutant)
                logging.debug('Candidate score: {:.5f}'.format(score_mutant))
                logging.debug(mutant)
            else:
                scores_gen.append(score_target)
                logging.debug('Candidate score: {:.5f}'.format(score_target))
                logging.debug(target)

            # A bit risky but the conditional should work as a safeguard
            if allscores[_tostring(population[j])] < best_score:

                if labs is None:
                    raise value_error("Oops, something went very wrong!")

                # best_res=allparams[_tostring(population[j])]
                best_res = labs, eps_opt, pj, mapping, keepfeat, decomp
                best_score = allscores[_tostring(population[j])]
                best_param = population[j]

        logging.debug('DE Generation ' + str(i + 1) + ' Results ')
        logging.debug('Average score: {:.5f}'.format(sum(scores_gen) / n_candidates))
        logging.debug('Best score: {:.5f}'.format(best_score))
        logging.debug('Best solution: ', best_param)

        if i > 1:
            if best_history[-3] - best_score < tol:
                logging.info('Score tolerance reached < {:2e}'.format(tol))
                break

    """ Reformat for optimization surface plitting. """

    # dumb format to improve
    allscores_list = [[], [], []]
    for key, value in allscores.items():
        allscores_list[2].append(1 - value)
        allscores_list[0].append(float(key.split('_')[0]))
        allscores_list[1].append(int(key.split('_')[1]))

    logging.info('Best solution: {:.2f}, {:3d}'.format(best_param[0],best_param[1]))
    
    return best_param, [1 - best_score]+list(best_res), \
           allscores_list
