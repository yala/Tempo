from oncopolicy.metrics.factory import RegisterMetric
import pdb
import numpy as np


@RegisterMetric("annualized_mammography_cost")
def annualized_mammography_cost(screen_progression, batch, args):
    '''
        Reward is negative average sceens per year.
        Note, this assumes all screens have unit cost 1 and are not MRI.
    '''
    time_spans = (np.max(screen_progression, axis=1) - np.min(screen_progression, axis=1)) / 2
    num_images =  np.array( [len(set(screen))-1 for screen in screen_progression ])
    cost = - (num_images / (time_spans +1e-9))
    return cost
