"""
Hyperparameter optimization for RACCOON
(Robust Adaptive Coarse-to-fine Clustering OptimizatiON)
F. Comitani     @2021
"""

# Please note: this implementation is basic but it works. 
# It needs to be improved badly.

import os
import shutil
import pickle
import random

import logging

import ray
from ray import tune
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, ExperimentPlateauStopper
from ray.tune.suggest.hyperopt import HyperOptSearch

class _htune_func(tune.Trainable):
        
    """ Define a custom trainable class. """

    def setup(self, config, obj_func, dataset):
        """ Setup the trainable object.
            
        Args:
            config (dict): dictionary with the parameters to test.
                It will be provided by ray tune.
            obj_func (func): objective function.
            dataset (dataframe or np matrix): the dataset to be passed
                to the objective function. 
        """
        
        self.x            = 0
        self.params       = list(config.values())
        self.score_target = -1
        self.labs         = None
        self.eps_opt      = 0
        self.pj           = None
        self.mapping      = None
        self.keepfeat     = []
        self.decomp       = None
        self.obj_func     = obj_func
        self.dataset      = dataset
    
    def step(self):  
        """ A single training step. 
            Evaluate the objective function.
        """

        # Please note the data needs to be explicitly input in the objective function
        # because I couldn't find an easy way to make ray see static variables coming 
        # from outside its own scope.
         
        self.score, self.labs, self.eps_opt, self.pj, self.mapping,\
                self.keepfeat, self.decomp = self.obj_func(self.params, 
                                                           dataset=self.dataset)

        self.x+=1

        return {"score": self.score}
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        """ Save checkpoint to disk.

        Args:
            tmp_checkpoint_dir (string): path to the temporary
                checkpoint directory.

        Returns:
            (string): full path to the saved checkpoint file.
        """

        # Couldn't find an easy way to store the trained maps and 
        # report them without filling up the memory.
        # This is a less then ideal workaround. Saving at every 
        # trial may save memory but it slows everything else down.

        with open(os.path.join(tmp_checkpoint_dir, 'model.pkl'),'wb') as chk:
            pickle.dump([self.score, self.labs, self.eps_opt, self.pj, self.mapping,\
                self.keepfeat, self.decomp], chk)

        return os.path.join(tmp_checkpoint_dir, 'model.pkl')


def _hyperparam_tune(obj_func_fun, hspace, dataset=None,
                       n_candidates=20, tol=1e-4, 
                       seed=None, searcher=None, outpath='./'):
    """ Hyperparameter tuning with Ray Tune.

    Args:
        obj_func_fun (function): objective function; takes a set of parameters to be optimized
            and returns a single float value.
        hspace (dictionary): hyperparameters space to explore.
        dataset (pandas dataframe or np matrix): the dataset to be used for the exploration.
        candidates (int): maximum number of candidate points in the hyperspace to explore.
        tol (float): solution improvement tolerance.
        seed (int): seed for the random numbers generator.
        searcher (tune suggest obj): a searcher algorithm from ray.tune.suggest, 
            if None, use Hyperopt (must have hyperopt installed, default None).

    Returns:
        (list of floats): list of best parameters.
        (list of objects): a list containing score, labels, clustering parameter,
            projected points, trained maps, filtered features and
            trained low-information filter from the best scoring model.
        (list of floats): a matrix containing all the explored models' parameters
            and their scores (useful for plotting the hyperspace).
    """

    if seed is not None:
        random.seed(seed)

    """ Reformat the hyperspace with tune. """

    hspace['ffrange'] = tune.uniform(min(hspace['ffrange']), max(hspace['ffrange']))
    hspace['nnrange'] = tune.loguniform(min(hspace['nnrange']), max(hspace['nnrange']))

    """ Set custom stopper, experiment plateau + maximum iteration. 
        The model is not trained iteratively so only one iteration per trial is needed.
        The search will be stopped if the score of the models reaches a plateau
        and maintains for n=patience iterations. """

    stopper  = CombinedStopper(
        ExperimentPlateauStopper('score', std=tol, top=3, mode="max", patience=3),
        MaximumIterationStopper(max_iter=1)
        )

    """ Set suggestion algorithm to guide the search. """

    if searcher is None:
        searcher = HyperOptSearch(metric="score", mode="max")

    """ Run tuning. """
    ray.init(include_dashboard=False)
    analysis = tune.run(
        tune.with_parameters(_htune_func, obj_func=obj_func_fun, dataset=dataset),
        mode                  = 'max',
        search_alg            = searcher,
        config                = hspace,
        stop                  = stopper,
        num_samples           = n_candidates,
        checkpoint_freq       = 0,
        keep_checkpoints_num  = 1, 
        checkpoint_score_attr = "score",
        checkpoint_at_end     = True,
        verbose               = 0,
        local_dir             = os.path.join(outpath, 'raytune_chk'))

    if len(analysis.results)<n_candidates:
        logging.info('Score tolerance reached < {:2e}'.format(tol))
    else:
        logging.info('Exausted all {:2d} candidates'.format(n_candidates))

    best_param = list(analysis.get_best_config(
        metric="score", mode="max").values())
    best_param = [best_param[0],int(best_param[1])]

    logging.info('Best solution: {:.2f}, {:d}'.format(best_param[0],best_param[1]))

    """ Recover best results, best parameters. """

    trial_logdir = analysis.get_best_logdir(metric="score", mode="max")
    with open(analysis.get_best_checkpoint(trial_logdir, metric="score", mode="max"), 'rb') as chk:
        best_res = pickle.load(chk)

    """ Clean up. """

    shutil.rmtree(os.path.join(outpath, 'raytune_chk'))

    return best_param, best_res,\
        analysis.results_df[['score','config.ffrange','config.nnrange']].values
