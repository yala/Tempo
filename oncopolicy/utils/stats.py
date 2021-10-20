import numpy as np
import tqdm
import sklearn.metrics
import warnings
import pdb
from p_tqdm import p_umap

RESAMPLE_FAILED_WARNING = "Resampling distrubution for estimator {} failed because of : {}"
CONFIDENCE_INTERVAL_EXCEPTION = "Cannot calculate confidence interval. Sampled {} times for found {} (target {}) valid samples."


def confidence_interval(emperical_distribution, confidence_interval=0.95, num_resamples=5000, estimator=np.mean, clusters=None):
    '''
    Estimates confidence interval of the mean of emperical_distribution
    using emperical bootstrap. Method details are available at:

    https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf

    -emperical_distribution: Emperical distributions for which we want to
    estimate the confidence interval of the mean
    -confidence_interval: Amount of probability mass interval should cover.
    -num_resamples: Amount of trails to use to estimate confidence interval.
    Should as large as computationally feasiable.

    if Clusters is not none, perform two stage bootstrap.
    '''
    emp_mean = estimator(emperical_distribution)
    num_samples = len(emperical_distribution)
    deltas = []
    num_tries = 0
    if clusters is not None:
        cluster_to_inds = {}
        for ind, cluster in enumerate(clusters):
            if cluster not in cluster_to_inds:
                cluster_to_inds[cluster] = []
            cluster_to_inds[cluster].append(ind)
        num_clusters = len(cluster_to_inds)
        cluster_ids = list(cluster_to_inds.keys())

    def get_delta(dummy_ind):
        if clusters is None:
            resample_ind = np.random.choice(a=range(len(emperical_distribution)),
                                    size=num_samples,
                                    replace=True
                                    )
        else:
            resample_clusters = np.random.choice(a=cluster_ids,
                                                size=num_clusters,
                                                replace=True)
            resample_ind = []
            for cluster in resample_clusters:
                resampled_inds_for_cluster = np.random.choice(a=cluster_to_inds[cluster],
                                                            size=len(cluster_to_inds[cluster]),
                                                            replace=True)
                resample_ind.extend(resampled_inds_for_cluster)

        resample_dist = [ emperical_distribution[ind] for ind in resample_ind]
        delta = estimator(resample_dist) - emp_mean
        return delta

    deltas = p_umap(get_delta, range(num_resamples))
    lower_bound, upper_bound = get_ci_bounds(emp_mean, deltas, confidence_interval, num_resamples)

    return emp_mean, (lower_bound, upper_bound)

def get_ci_bounds(mean, deltas, confidence_interval, num_resamples):
    '''
    Returns the lower and upper bounds of the confidence interval.
    mean : the empirical mean
    deltas : deltas from the empirical mean from different samples
    '''
    deltas = np.sort(deltas)
    index_offset = int((1 - confidence_interval)/2. * num_resamples)
    lower_delta, upper_delta = deltas[index_offset], deltas[-index_offset]
    lower_bound, upper_bound = mean - upper_delta, mean - lower_delta
    return lower_bound, upper_bound
