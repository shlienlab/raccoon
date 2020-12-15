"""
Basic differential evolution implementation for RACCOON (Recursive Algorithm for Coarse-to-fine Clustering OptimizatiON)
F. Comitani     @2019

Based on
https://nathanrooy.github.io/posts/2017-08-27/simple-differential-evolution-with-python

Storn, R.; Price, K. (1997). "Differential evolution - a simple and efficient heuristic 
for global optimization over continuous spaces". Journal of Global Optimization. 
11 (4): 341–359. doi:10.1023/A:1008202821328.

"""

import random
import numpy as np
import logging
from decimal import Decimal, ROUND_HALF_UP
from math import nan

def _clamp(x, minVal, maxVal):

    """ Force a number between bounds.

    Args:
        x (float): Input value to be clamped.
        minVal (float): Lower bound.
        maxVal (float): Upper bound.

    Returns:
        (float): Clamped value.
    """

    return max([min([maxVal, x]), minVal])

def _tostring(x):

    """ Conbine a list of numbers into a single string with underscore as separator.

    Args:
        x (list of floats): List of numbers to combine.

    Returns:
        (str): Combined string.
    """

    string=''
    for xx in x:
        string=string+str(xx)+'_'
    return string[:-1]

def _differentialEvolution(lossFun, bounds, integers=None, popsize=10, mutation=0.6, recombination=0.7, maxiter=20, tol=1e-4, seed=None):

    """ Basic Differential Evolution implementation.

    Args:
        lossFun (function): Objective function; takes a set of parameters to be optimized and returns a single float value.
        bounds (tuple): Minimum and maximum boundaries for the parameters to optimize.
        integers (list of booleans or None): List with information on which parameters are integers,
            if None (default) treat every parameter as float. 
        popsize (int): Size of the candidate solutions population.
        mutation (float): Scaling factor for the mutation step.
        recombination (float): Recombination (crossover) rate.
        maxiter (float): maximum number of generations.
        tol (float): solution improvement tolerance, 
            if after 3 generations the best solution is not improved by at least this value, stop the iteration.
        seed (int): seed for the random numbers generator.

    Returns:
        (list of floats): List of best parameters.
    """
    
    if seed is not None:
        random.seed(seed)

    """ Randomly initialize candidated solutions. """

    allscores={}

    #TO DO: Fix redundancy in saving the parameters
    #allparams={}

    population=[]
    for i in range(popsize):
        indv=[]
        for j in range(len(bounds)):
            if integers is not None and integers[j]:
                indv.append(random.randint(bounds[j][0],bounds[j][1]))
            else:
                indv.append(random.uniform(bounds[j][0],bounds[j][1]))

        population.append(indv)
    
    
    """ Run epochs. """

    bestRes=[nan]*7
    bestScore=1e100
    bestHistory=[]

    for i in range(maxiter):

        logging.debug('DE generation: {:d}'.format(i+1))

        bestHistory.append(bestScore)

        scoresGen=[] 

        """ Cycle through the whole population. """

        for j in range(popsize):

            """ Create hybrid. """
            
            candidIx=list(range(popsize))
            candidIx.remove(j)
            candidIx=random.sample(candidIx, 3)

            agents=[population[candidIx[0]],
                     population[candidIx[1]],
                     population[candidIx[2]]]  
            
            target=population[j]  

            hybrid=[x-y for x, y in zip(agents[1], agents[2])]
            hybrid=[x + mutation * y for x, y in zip(agents[0], hybrid)]
            hybrid=[_clamp(hybrid[k],bounds[k][0],bounds[k][1]) for k in range(len(hybrid))]


            """ Make sure integer parameters stay as such. """

            for k in range(len(bounds)):
                    if integers is not None and integers[k]:
                        hybrid[k]=int(Decimal(str(hybrid[k])).to_integral_value(rounding=ROUND_HALF_UP))
            
            """ Recombine target solution. """

            mutant=[]
            for k in range(len(target)):
                if random.random()<=recombination:
                    mutant.append(hybrid[k])
                else:
                    mutant.append(target[k])
                    
            """ Select the solution with best score,
                if candidate's score has been evaluated before, skip and use old score. """

            #Note: keep the results for the best candidate so we don't have to recalculate them.

            if _tostring(target) not in allscores:
                scoreTarget, labs, epsOpt, pj, mapping, keepfeat, decomp, parmvals = lossFun(target)
                allscores[_tostring(target)]=scoreTarget
                #allparams[_tostring(target)]=[labs, epsOpt, pj, keepfeat, decomp, parmvals]
            else:
                logging.debug('Candidate found in store')
                scoreTarget=allscores[_tostring(target)]
                labs=None

            if _tostring(mutant) not in allscores:
                scoreMutant, labsM, epsOptM, pjM, mappingM, keepfeatM, decompM, parmvalsM = lossFun(mutant)
                allscores[_tostring(mutant)]=scoreMutant
                #allparams[_tostring(mutant)]=[labsM, epsOptM, pjM, keepfeatM, decompM, parmvalsM]
            else:
                logging.debug('Candidate found in store')
                scoreMutant=allscores[_tostring(mutant)]
                labsM=None

            if scoreMutant < scoreTarget:
                population[j]=mutant
                labs, epsOpt, pj, mapping, keepfeat, decomp, parmvals = labsM, epsOptM, pjM, mappingM, keepfeatM, decompM, parmvalsM
                scoresGen.append(scoreMutant)
                logging.debug('Candidate score: {:.5f}'.format(scoreMutant))  
                logging.debug(mutant)
            else:
                scoresGen.append(scoreTarget)
                logging.debug('Candidate score: {:.5f}'.format(scoreTarget))        
                logging.debug(target)
        
            """ A bit risky but the conditional should work as a safeguard """

            if allscores[_tostring(population[j])]<bestScore:
                
                if labs is None:
                    raise ValueError("Oops, something went very wrong!")
                
                #bestRes=allparams[_tostring(population[j])]
                bestRes=labs, epsOpt, pj, mapping, keepfeat, decomp, parmvals
                bestScore=allscores[_tostring(population[j])]
                bestParam=population[j]     

        logging.debug('DE Generation '+str(i+1)+' Results ')
        logging.debug('Average score: {:.5f}'.format(sum(scoresGen)/popsize))
        logging.debug('Best score: {:.5f}'.format(bestScore))
        logging.debug('Best solution: ')
        logging.debug(bestParam)

        if i>1:
            if bestHistory[-3]-bestScore<tol:
                logging.info('Tolerance reached < {:2e}'.format(tol))
                break

    return 1-bestScore, bestRes[0], bestRes[1], bestParam[1], bestRes[2], bestParam[0], bestRes[3], bestRes[4], bestRes[5], bestRes[6]
