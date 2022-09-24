import logging
import random
import numpy as np
from aslib_scenario.aslib_scenario import ASlibScenario
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import wandb


import pdb

class SATzilla11(nn.Module):
    """ 
    Implementation of SATzilla's internal algorithm selector. 

    Note, however, that this implementation does not account for SATzilla's many other steps, 
    thus does not account for presolving or employing a backup solver.    
    """

    def __init__(self, max_fidelity, device = 'cpu'):
        super(SATzilla11, self).__init__()
        
        self._name = 'satzilla-11'
        self._models = {}
        self.max_fidelity = max_fidelity
        self.device = device
        
        # used to break ties during the predictions phase
        self._rand = random.Random(0)
        
        # TODO Initiate the classifier using hydra
        self.rfc_kwargs = {
            'n_estimators': 100,
            'max_features': 'log2', 
            'n_jobs': 5, 
            'random_state': 0

        }

    def forward(self, dataset_meta_features, fidelity):

        

        
        self.fit(dataset_meta_features.numpy()[0], fidelity.numpy()[0])
        
        self.predict(dataset_meta_features.numpy()[0])


    def fit(self, dataset_meta_features, fidelity):

        self._num_algorithms = np.shape(fidelity)[-1]
        self._algo_ids = np.arange(0,self._num_algorithms,1)

        features = dataset_meta_features.numpy()[0]
        performances = fidelity.numpy()[0]

        # create and fit rfcs' for all pairwise comparisons between two algorithms
        self._pairwise_indices = [(i, j) for i in range(self._num_algorithms) for j in range(i + 1, self._num_algorithms)]


        for (i, j) in tqdm(self._pairwise_indices):
            # determine pairwise target, initialize models and fit each RFC wrt. instance weights
            pair_target = self._get_pairwise_target((i, j), performances)
            sample_weights = self._compute_sample_weights((i, j), performances)

            # account for nan values (i.e. ignore pairwise comparisons that involve an algorithm run violating
            # the cutoff if the config specifies 'ignore_censored'), hence set all respective weights to 0
            sample_weights = np.nan_to_num(sample_weights)

            # TODO: how to set the remaining hyperparameters? 
            self._models[(i, j)] = RandomForestClassifier(**self.rfc_kwargs)     # TODO set random state to the torch seed
            self._models[(i, j)].fit(features, pair_target, sample_weight=sample_weights)


    def predict(self, dataset_meta_features, fidelity):
        
        batch_features = dataset_meta_features.numpy()[0]
        
        selections = []
        # Selection an algorithm per task in the test-set
        for features in batch_features:
                       
            assert(features.ndim == 1), '`features` must be one dimensional'
            features = np.expand_dims(features, axis=0)

            # compute the most voted algorithms for the given instance
            predictions = {(i, j): rfc.predict(features).item()
                        for (i, j), rfc in self._models.items()}
 
            counter = Counter(predictions.values())
            
            for id in self._algo_ids:
                if id not in counter.keys():
                    counter[id] = 0
                
            
            # TODO revisit the tie breaking procedure
            # ranking = []

            
            # for i in range(len(counter.most_common())):
                
            #     key, val  = counter.most_common()[i]

            #     if counter.most_common()[i+1][1] == val:

            #         ties  = [key]

            #         k = i+1
            #         tie  = True
            #         while tie:
            #             ties.append(counter.most_common()[k][0])

            #             k += 1 
            #             if counter.most_common()[k][1] != val:
            #                 tie = False

            #         print(ties)        
            #         [ranking.append(a) for a,_ in self._break_ties(ties, predictions)]
            #     else:
            ranking = [key for key, _ in counter.most_common()]            
            
            selections.append(ranking)

        return torch.Tensor(selections)

    def _get_pairwise_target(self, pair, performances):
        i, j = pair

        # label target as i if the utility of algorithm i is <= the utility of algorithm j,
        # otherwise label it j
        pair_target = np.full(performances.shape[0], fill_value=i)        
        pair_target[performances[:, i] > performances[:, j]] = j

        return pair_target

    def _compute_sample_weights(self, pair, performances):
        i, j = pair

        # sample weights are proportional to the difference in performances of all algorithms on
        # the dataset
        weights = np.absolute(performances[:, i] - performances[:, j])

        return weights

    # NOTE better method to break teis? 
    def _break_ties(self, tied, predictions):
        '''
        Break ties between algorithms by checking the ones predicted by the algorithms in the ties
        '''
        indices = [(i, j) for (i, j) in self._pairwise_indices if i in tied and j in tied]
        print(indices)

        pred = {(i, j): predictions[(i, j)] for (i, j) in indices}
        
        counter = Counter(pred.values())

        print('counter', counter)
        pdb.set_trace()
        
        return counter.most_common()


if __name__ == "__main__":

    x = SATzilla11()

    print('Hallelujah')
