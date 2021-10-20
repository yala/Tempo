import pickle
import torch
import pdb

NO_METRIC_ERR = "Metric {} not in METRIC_REGISTRY! Available metrics are {}"
METRIC_REGISTRY = {}
alt_metrics = ['main_loss', 'mse', 'abs_five_year_error']

def RegisterMetric(metric):
    """Registers a dataset."""

    def decorator(f):
        METRIC_REGISTRY[metric] = f
        return f

    return decorator

# Depending on arg, build dataset
def compute_reward(screen_progression, batch, args):
    batch_size = batch['last_neg_time_step'].size()[0]
    metric_rewards = []
    for metric in args.metrics:
        metric_rewards.append(METRIC_REGISTRY[metric](screen_progression, batch, args))
    reward_tensor = torch.Tensor(metric_rewards).to(args.device)
    return reward_tensor

def get_metric_keys(screen_only = False):
    base_metrics =  list(METRIC_REGISTRY.keys())
    if not screen_only:
        base_metrics.extend( alt_metrics)
    return base_metrics

def get_metrics_with_cis():
    base_metrics =  list(METRIC_REGISTRY.keys()) + alt_metrics
    metrics_with_cis = []
    for metric in base_metrics:
        metrics_with_cis.extend( [ metric, "{}_lower_95".format(metric), "{}_upper_95".format(metric)])
    return metrics_with_cis
