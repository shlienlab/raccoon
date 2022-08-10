"""
Tree-structured Parzen Estimators optimization for RACCOON
F. Comitani     @2021
"""

import os

import logging

from math import inf
import operator

import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

class Objective:
    """ Objective function class for Optuna."""

    def __init__(self, bounds, obj_func):
        """ Initialize the objective object.
        
        Args:
            obj_func (function): objective function; takes a set of parameters to be optimized
                and returns a single float value.
            bounds (tuple): minimum and maximum boundaries for the parameters to optimize.
        """

        self.best_model = None
        self._model = None
        self.bounds = bounds
        self.obj_func = obj_func

    def __call__(self, trial):
        """ Runs a single istance of the ojbective function evaluation.

        Args:
            trial (optuna.Trial): the current trial.

        Returns:
            score (float): the objective function result on the current trial.
        """

        score, labs, eps_opt, pj, mapping,\
            keepfeat, decomp = \
            self.obj_func([trial.suggest_uniform("ffparm", self.bounds[0][0], self.bounds[0][1]),
                          trial.suggest_int("nnrange", self.bounds[1][0], self.bounds[1][1], log=True)])

        self._model = [1 - score, labs, eps_opt, pj, mapping,\
            keepfeat, decomp]
        
        logging.debug('Score: {:.5f}'.format(1-score))
        
        return score

    def callback(self, study, trial):
        """ Stores the best results.
        
        Args:
            study (optuna.Study): the study to interrupt.
            trial (optuna.Trial): the current trial.    
        """

        if study.best_trial == trial:
            self.best_model = self._model
            
class EarlyStoppingCallback(object):
    """ Early stopping callback for Optuna. """

    def __init__(self, patience = 5, tolerance = 1e-4, direction = "minimize"):
        """ Initialize early stopping.

        Args:
            patience (int): number of rounds to wait after reaching the plateau
                before stopping the study (default 5).
            tolerance (float): solution improvement tolerance (default 1e-4).
            direction (str): Direction of the optimization, it can be
                either "minimize" or "minimize" in accordance
                to Optuna's format (default "minimize").
        """

        self.patience = patience
        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self.tolerance = -tolerance
            self._score = inf
        elif direction == "minimize":
            self._operator = operator.gt
            self.tolerance = tolerance
            self._score = -inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study, trial):
        """ Checks if the study needs to be stopped or can continue.
        
        Args:
            study (optuna.Study): the study to interrupt.
            trial (optuna.Trial): the current trial.    
        """

        if self._operator(study.best_value, self._score+self.tolerance):
            self._iter = 0
            self._score = study.best_value
        else:
            #elif self._score <=1:
            self._iter += 1

        if self._iter >= self.patience - 1:
            study.stop()

def _optuna_tpe(obj_func, bounds, n_candidates=20, 
                       patience = 5, tol=1e-4, 
                       seed=None):
    """ Tree-structured Parzen Estimators optimization with Optuna.

    Args:
        obj_func (function): objective function; takes a set of parameters to be optimized
            and returns a single float value.
        bounds (tuple): minimum and maximum boundaries for the parameters to optimize.
        candidates (int): maximum number of candidate points in the hyperspace 
            to explore (default 20).
        patience (int): number of rounds to wait after reaching the plateau
            before stopping the study (default 5).
        tol (float): solution improvement tolerance (default 1e-4).
        seed (int): seed for the random numbers generator (default None).

    Returns:
        (tuple (list of floats, list of objects, list of floats)): tuple containing
            the list of best parameters; 
            a list containing score, labels, clustering parameter,
            projected points, trained maps, filtered features and
            trained low-information filter from the best scoring model;
            a matrix containing all the explored models' parameters
            and their scores (useful for plotting the hyperspace).
    """

    """ Set Objective function. """
    
    objective = Objective(bounds, obj_func)
    
    """ Set Early stopping. """

    early_stopping = EarlyStoppingCallback(patience=patience, tolerance=tol, 
                                           direction="minimize")

    """ Run Optuna study. """

    study = optuna.create_study(sampler=TPESampler(seed=seed), direction="minimize")
    study.optimize(objective, n_trials=n_candidates, 
            callbacks=[objective.callback, early_stopping])

    if len(study.trials) < n_candidates:
        logging.info('Score tolerance reached < {:2e}'.format(tol))
    else:
        logging.info('Exausted all {:2d} candidates'.format(n_candidates))

    """ Recover best parameters. """

    best_param = list(study.best_trial.params.values())
    best_param = [best_param[0],int(best_param[1])]

    logging.info('Best solution: {:.2f}, {:d}'.format(best_param[0],best_param[1]))
    return best_param, objective.best_model,\
        list(map(list, zip(*[list(x.params.values())+[1-x.values[0]] for x in study.trials])))
